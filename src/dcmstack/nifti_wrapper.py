"""Provide wrapper for a nibabel Nifti image with a meta data extension

Also provides conversion from single DICOM to equivalent Nifti plus meta data
"""
import itertools
import warnings
from typing import List, Sequence, Optional, Union

import numpy as np
import nibabel as nb
from nibabel.spatialimages import HeaderDataError
from nibabel.nifti1 import Nifti1Image, Nifti1Extension
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from nibabel.nicom.dicomwrappers import wrapper_from_data

from .dcmmeta import DcmMetaExtension, InvalidExtensionError, dcm_meta_ecode


class MissingExtensionError(Exception):
    '''Exception denoting that there is no DcmMetaExtension in the Nifti header.
    '''
    def __str__(self):
        return 'No dcmmeta extension found.'


def patch_dcm_ds_is(dcm):
    '''Convert all elements in `dcm` with VR of 'DS' or 'IS' to floats and ints.
    This is a hackish work around for the backwards incompatibility of pydicom
    0.9.7 and should not be needed once nibabel is updated.
    '''
    for elem in dcm:
        if elem.VM == 1:
            if elem.VR in ('DS', 'IS'):
                if elem.value == '':
                    continue
                if elem.VR == 'DS':
                    elem.VR = 'FD'
                    elem.value = float(elem.value)
                else:
                    elem.VR = 'SL'
                    elem.value = int(elem.value)
        else:
            if elem.VR in ('DS', 'IS'):
                if elem.value == '':
                    continue
                if elem.VR == 'DS':
                    elem.VR = 'FD'
                    elem.value = [float(val) for val in elem.value]
                else:
                    elem.VR = 'SL'
                    elem.value = [int(val) for val in elem.value]


def gen_simplified_sequences(meta_dict):
    """Get rid of useless nesting of meta data from multiframe DICOM"""
    for k, v in meta_dict.items():
        if isinstance(v, list):
            if len(v) == 0:
                continue
            if len(v) == 1:
                for sub_key, sub_val in v[0].items():
                    yield sub_key, sub_val
                continue
        yield k, v
    

def _get_fg_const_and_varying(fg_seqs):
    varying = {}
    for idx, fg_seq in enumerate(fg_seqs):
        for k, v in gen_simplified_sequences(fg_seq):
            if k not in varying:
                if idx == 0:
                    varying[k] = [v]
                else:
                    varying[k] = [None] * idx
                    varying[k].append(v)
            else:
                n_vals = len(varying[k])
                if n_vals != idx:
                    varying[k] += [None] * (idx - n_vals)
                varying[k].append(v)
    const = {}
    for k, vals in varying.items():
        if len(vals) != len(fg_seqs):
            vals += [None] * (len(fg_seqs) - len(vals))
        if all(x == vals[0] for x in vals):
            const[k] = vals[0]
    for k in const:
        del varying[k]
    return const, varying

class NiftiWrapper(object):
    '''Wraps a Nifti1Image object containing a DcmMeta header extension.
    Provides access to the meta data and the ability to split or merge the
    data array while updating the meta data.

    Parameters
    ----------
    nii_img : nibabel.nifti1.Nifti1Image
        The Nifti1Image to wrap.

    make_empty : bool
        If True an empty DcmMetaExtension will be created if none is found.

    Raises
    ------
    MissingExtensionError
        No valid DcmMetaExtension was found.

    ValueError
        More than one valid DcmMetaExtension was found.
    '''

    def __init__(self, nii_img: Nifti1Image, make_empty: bool = False):
        self.nii_img = nii_img
        hdr = nii_img.header
        ext = None
        self.meta_ext: Nifti1Extension = None
        for extension in hdr.extensions:
            if extension.get_code() == dcm_meta_ecode:
                try:
                    extension.check_valid()
                except InvalidExtensionError as e:
                    print("Found candidate extension, but invalid: %s" % e)
                else:
                    if not ext is None:
                        raise ValueError('More than one valid DcmMeta '
                                         'extension found.')
                    ext = extension
        if ext is None:
            if make_empty:
                slice_dim = hdr.get_dim_info()[2]
                ext = DcmMetaExtension.make_empty(
                    self.nii_img.shape,
                    hdr.get_best_affine(),
                    None,
                    slice_dim
                )
                hdr.extensions.append(ext)
            else:
                raise MissingExtensionError
        self.meta_ext: Nifti1Extension = ext

    def __getitem__(self, key):
        '''Get the value for the given meta data key. Only considers meta data
        that is globally constant. To access varying meta data you must use the
        method 'get_meta'.'''
        return self.meta_ext.get_class_dict(('global', 'const'))[key]

    def meta_valid(self, classification):
        '''Return true if the meta data with the given classification appears
        to be valid for the wrapped Nifti image. Considers the shape and
        orientation of the image and the meta data extension.'''
        if classification == ('global', 'const'):
            return True

        img_shape = self.nii_img.shape
        meta_shape = self.meta_ext.shape
        if classification == ('vector', 'samples'):
            return meta_shape[4:] == img_shape[4:]
        if classification == ('time', 'samples'):
            return meta_shape[3:] == img_shape[3:]

        hdr = self.nii_img.header
        if self.meta_ext.n_slices != hdr.get_n_slices():
            return False

        slice_dim = hdr.get_dim_info()[2]
        slice_dir = self.nii_img.affine[slice_dim, :3]
        slices_aligned = np.allclose(slice_dir,
                                     self.meta_ext.slice_normal,
                                     atol=1e-6)

        if classification == ('time', 'slices'):
            return slices_aligned
        if classification == ('vector', 'slices'):
            return meta_shape[3] == img_shape[3] and slices_aligned
        if classification == ('global', 'slices'):
            return meta_shape[3:] == img_shape[3:] and slices_aligned

    def get_meta(self, key, index=None, default=None):
        '''Return the meta data value for the provided `key`.

        Parameters
        ----------
        key : str
            The meta data key.

        index : tuple
            The voxel index we are interested in.

        default
            This will be returned if the meta data for `key` is not found.

        Returns
        -------
        value
            The meta data value for the given `key` (and optionally `index`)

        Notes
        -----
        The per-sample and per-slice meta data will only be considered if the
        `samples_valid` and `slices_valid` methods return True (respectively),
        and an `index` is specified.
        '''
        #Get the value(s) and classification for the key
        values, classes = self.meta_ext.get_values_and_class(key)
        if classes is None:
            return default

        #Check if the value is constant
        if classes == ('global', 'const'):
            return values

        #Check if the classification is valid
        if not self.meta_valid(classes):
            return default

        #If an index is provided check the varying values
        if not index is None:
            #Test if the index is valid
            shape = self.nii_img.shape
            if len(index) != len(shape):
                raise IndexError('Incorrect number of indices.')
            for dim, ind_val in enumerate(index):
                if not 0 <= ind_val < shape[dim]:
                    raise IndexError('Index is out of bounds.')

            #First try per time/vector sample values
            if classes == ('time', 'samples'):
                return values[index[3]]
            if classes == ('vector', 'samples'):
                return values[index[4]]

            #Finally, if aligned, try per-slice values
            slice_dim = self.nii_img.header.get_dim_info()[2]
            n_slices = shape[slice_dim]
            if classes == ('global', 'slices'):
                val_idx = index[slice_dim]
                for count, idx_val in enumerate(index[3:]):
                    val_idx += idx_val * n_slices
                    n_slices *= shape[count+3]
                return values[val_idx]
            elif classes == ('time', 'slices'):
                val_idx = index[slice_dim]
                return values[val_idx]
            elif classes == ('vector', 'slices'):
                val_idx = index[slice_dim]
                val_idx += index[3]*n_slices
                return values[val_idx]

        return default

    def remove_extension(self):
        '''Remove the DcmMetaExtension from the header of nii_img. The
        attribute `meta_ext` will still point to the extension.'''
        hdr = self.nii_img.header
        target_idx = None
        for idx, ext in enumerate(hdr.extensions):
            if id(ext) == id(self.meta_ext):
                target_idx = idx
                break
        else:
            raise IndexError('Extension not found in header')
        del hdr.extensions[target_idx]
        # Nifti1Image.update_header will increase this if necessary
        hdr['vox_offset'] = 0

    def replace_extension(self, dcmmeta_ext: Nifti1Extension):
        '''Replace the DcmMetaExtension.

        Parameters
        ----------
        dcmmeta_ext : DcmMetaExtension
            The new DcmMetaExtension.

        '''
        self.remove_extension()
        self.nii_img.header.extensions.append(dcmmeta_ext)
        self.meta_ext = dcmmeta_ext

    def split(self, dim=None):
        '''Generate splits of the array and meta data along the specified
        dimension.

        Parameters
        ----------
        dim : int
            The dimension to split the voxel array along. If None it will
            prefer the vector, then time, then slice dimensions.

        Returns
        -------
        result
            Generator which yields a NiftiWrapper result for each index
            along `dim`.

        '''
        shape = self.nii_img.shape
        data = np.asanyarray(self.nii_img.dataobj)
        header = self.nii_img.header
        slice_dim = header.get_dim_info()[2]

        #If dim is None, choose the vector/time/slice dim in that order
        if dim is None:
            dim = len(shape) - 1
            if dim == 2:
                if slice_dim is None:
                    raise ValueError("Slice dimension is not known")
                dim = slice_dim

        #If we are splitting on a spatial dimension, we need to update the
        #translation
        trans_update = None
        if dim < 3:
            trans_update = header.get_best_affine()[:3, dim]

        split_hdr = header.copy()
        slices = [slice(None)] * len(shape)
        for idx in range(shape[dim]):
            #Grab the split data, get rid of trailing singular dimensions
            if dim >= 3 and dim == len(shape) - 1:
                slices[dim] = idx
            else:
                slices[dim] = slice(idx, idx+1)

            split_data = data[tuple(slices)].copy()

            #Update the translation in any affines if needed
            if not trans_update is None and idx != 0:
                qform = split_hdr.get_qform()
                if not qform is None:
                    qform[:3, 3] += trans_update
                    split_hdr.set_qform(qform)
                sform = split_hdr.get_sform()
                if not sform is None:
                    sform[:3, 3] += trans_update
                    split_hdr.set_sform(sform)

            #Create the initial Nifti1Image object
            split_nii = Nifti1Image(
                split_data,
                split_hdr.get_best_affine(),
                header=split_hdr
            )

            #Replace the meta data with the appropriate subset
            meta_dim = dim
            if dim == slice_dim:
                meta_dim = self.meta_ext.slice_dim
            split_meta = self.meta_ext.get_subset(meta_dim, idx)
            result = NiftiWrapper(split_nii)
            result.replace_extension(split_meta)

            yield result

    def to_filename(self, out_path):
        '''Write out the wrapped Nifti to a file

        Parameters
        ----------
        out_path : str
            The path to write out the file to

        Notes
        -----
        Will check that the DcmMetaExtension is valid before writing the file.
        '''
        self.meta_ext.check_valid()
        self.nii_img.to_filename(out_path)

    @classmethod
    def from_filename(klass, path):
        '''Create a NiftiWrapper from a file.

        Parameters
        ----------
        path : str
            The path to the Nifti file to load.
        '''
        return klass(nb.load(path))

    @classmethod
    def from_dicom_wrapper(klass, dcm_wrp, meta_dict=None):
        '''Create a NiftiWrapper from a nibabel DicomWrapper.

        Parameters
        ----------
        dcm_wrap : nicom.dicomwrappers.DicomWrapper
            The dataset to convert into a NiftiWrapper.

        meta_dict : dict
            An optional dictionary of meta data extracted from `dcm_data`. See
            the `extract` module for generating this dict.
        '''
        shape = dcm_wrp.image_shape
        n_dims = len(shape)
        if n_dims > 4:
            raise ValueError("5D+ multiframe not supported")
        # Figure out any data rescaling
        scale_factors = dcm_wrp.scale_factors
        if dcm_wrp.is_multiframe and scale_factors.shape[1] > 1:
            slope, inter = 1, 0
            data = dcm_wrp.get_data()
        else:
            slope, inter = scale_factors[0, :]
            data = dcm_wrp.get_unscaled_data()
        # The Nifti patient space flips the x and y directions
        affine = np.dot(np.diag([-1., -1., 1., 1.]), dcm_wrp.affine)
        n_vols = 1
        if n_dims == 2:
            data = data.reshape(data.shape + (1,))
            slices_per_vol = 1
        elif n_dims >= 3:
            data = np.squeeze(data)
            slices_per_vol = shape[2]
        if n_dims == 4:
            n_vols = shape[3]
        # Create the nifti image and set header data
        nii_img = Nifti1Image(data, affine)
        hdr = nii_img.header
        if (slope, inter) != (1.0, 0.0):
            hdr.set_slope_inter(slope, inter)
        hdr.set_xyzt_units('mm', 'sec')
        # Determine phase encoding direction, set dimension info in header
        phase_info = None
        if dcm_wrp.is_multiframe:
            phase_info = dcm_wrp.shared.get("MRFOVGeometrySequence")
            if phase_info is None and "MRFOVGeometrySequence" in dcm_wrp.frames[0]:
                phase_info = [f.get("MRFOVGeometrySequence")[0] for f in dcm_wrp.frames]
        if phase_info is None:
            phase_info = [dcm_wrp]
        phase_dirs = set(d.get('InPlanePhaseEncodingDirection') for d in phase_info)
        if len(phase_dirs) > 1:
            phase_dir = None
        else:
            phase_dir = phase_dirs.pop()
        dim_info = {'freq' : None, 'phase' : None, 'slice' : 2}
        if phase_dir:
            if phase_dir == 'ROW':
                dim_info['phase'], dim_info['freq'] = 1, 0
            else:
                dim_info['phase'], dim_info['freq'] = 0, 1
        hdr.set_dim_info(**dim_info)
        # Create result and embed any provided meta data
        result = klass(nii_img, make_empty=True)
        result.meta_ext.reorient_transform = np.eye(4)
        if meta_dict:
            if not dcm_wrp.is_multiframe:
                global_meta = result.meta_ext.get_class_dict(('global', 'const'))
                global_meta.update(meta_dict)
                if dcm_wrp.is_mosaic:
                    # For mosaic images we move a few elems that provide per-slice meta
                    slice_meta = result.meta_ext.get_class_dict(('global', 'slices'))
                    for key in (
                        "SIEMENS_MR_HEADER.MosaicRefAcqTimes",
                        "CsaImage.MosaicRefAcqTimes",
                        "SourceImageSequence",
                    ):
                        vals = global_meta.get(key)
                        if vals is not None:
                            slice_meta[key] = vals
                            del global_meta[key]
            else:
                # Unpack and sort meta data from multiframe file that varies
                global_meta = meta_dict.copy()
                del global_meta["SharedFunctionalGroupsSequence"]
                del global_meta["PerFrameFunctionalGroupsSequence"]
                assert len(meta_dict["SharedFunctionalGroupsSequence"]) == 1
                for k, v in gen_simplified_sequences(
                    meta_dict["SharedFunctionalGroupsSequence"][0]
                ):
                    if k in global_meta:
                        if global_meta[k] == v:
                            continue
                        k = f"Shared.{k}"
                    global_meta[k] = v
                fg_seqs = meta_dict.get("PerFrameFunctionalGroupsSequence")
                sorted_indices = dcm_wrp.frame_order
                fg_keys = set()
                slice_keys = set()
                vol_results = []
                for vol_idx in range(n_vols):
                    start = vol_idx * slices_per_vol
                    end = start + slices_per_vol
                    vol_res = _get_fg_const_and_varying(
                        [fg_seqs[x] for x in sorted_indices[start:end]]
                    )
                    vol_results.append(vol_res)
                    for res in vol_res:
                        for k in res:
                            fg_keys.add(k)
                    for k in vol_res[1]:
                        slice_keys.add(k)
                global_slices = result.meta_ext.get_class_dict(('global', 'slices'))
                if n_vols > 1:
                    for vconst, vslices in vol_results:
                        for k in fg_keys:
                            if k not in vconst and k not in vslices:
                                vconst[k] = None
                        for k in slice_keys:
                            if k in vconst:
                                vslices[k] = [vconst[k]] * slices_per_vol
                                del vconst[k]
                    time_samples = result.meta_ext.get_class_dict(('time', 'samples'))
                    time_slices = result.meta_ext.get_class_dict(('time', 'slices'))
                    for k, first_val in vol_results[0][0].items():
                        dest_key = k if k not in global_meta else f"PerFrame.{k}"
                        vals = [vconst[k] for vconst, _ in vol_results]
                        if all(v == first_val for v in vals):
                            if k in global_meta and global_meta[k] == first_val:
                                continue
                            global_meta[dest_key] = first_val
                        else:
                            time_samples[dest_key] = vals
                    for k, first_val in vol_results[0][1].items():
                        dest_key = k if k not in global_meta else f"PerFrame.{k}"
                        vals = [vslices[k] for _, vslices in vol_results]
                        if all(v == first_val for v in vals):
                            time_slices[dest_key] = first_val
                        else:
                            global_slices[dest_key] = list(itertools.chain(*vals))
                else:
                    vconst, vslices = vol_results[0]
                    for k, val in vconst.items():
                        if k in global_meta:
                            if global_meta[k] != val:
                                global_meta[f"PerFrame.{k}"] = val
                        else:
                            global_meta[k] = val
                    for k, vals in vslices.items():
                        if k in global_meta:
                            global_slices[f"PerFrame.{k}"] = vals
                        else:
                            global_slices[k] = vals
                result.meta_ext.get_class_dict(('global', 'const')).update(global_meta)
        return result

    @classmethod
    def from_dicom(klass, dcm_data, meta_dict=None):
        '''Create a NiftiWrapper from a single DICOM dataset.

        Parameters
        ----------
        dcm_data : dicom.dataset.Dataset
            The DICOM dataset to convert into a NiftiWrapper.

        meta_dict : dict
            An optional dictionary of meta data extracted from `dcm_data`. See
            the `extract` module for generating this dict.

        '''
        dcm_wrp = wrapper_from_data(dcm_data)
        return klass.from_dicom_wrapper(dcm_wrp, meta_dict)

    @classmethod
    def from_sequence(
        klass,
        seq: Sequence["NiftiWrapper"],
        dim: Optional[int] = None,
        dim_key: Optional[str] = None,
    ) -> "NiftiWrapper":
        '''Create a NiftiWrapper by joining a sequence of NiftiWrapper objects
        along the given dimension.

        Parameters
        ----------
        seq : sequence
            The sequence of NiftiWrapper objects.

        dim : int
            The dimension to join the NiftiWrapper objects along. If None,
            2D inputs will become 3D, 3D inputs will become 4D, and 4D inputs
            will become 5D.

        Returns
        -------
        result : NiftiWrapper
            The merged NiftiWrapper with updated meta data.
        '''
        n_inputs = len(seq)
        first_input = seq[0]
        first_nii = first_input.nii_img
        first_hdr = first_nii.header
        shape = first_nii.shape
        affine = first_nii.affine.copy()
        #If dim is None, choose a sane default
        if dim is None:
            if len(shape) == 3:
                singular_dim = None
                for dim_idx, dim_size in enumerate(shape):
                    if dim_size == 1:
                        singular_dim = dim_idx
                if singular_dim is None:
                    dim = 3
                else:
                    dim = singular_dim
            if len(shape) == 4:
                dim = 4
        else:
            if not 0 <= dim < 5:
                raise ValueError("The argument 'dim' must be in the range "
                                 "[0, 5).")
            if dim < len(shape) and shape[dim] != 1:
                raise ValueError('The dimension must be singular or not exist')
        assert dim is not None
        #Pull out the three axes vectors for validation of other input affines
        axes = []
        for axis_idx in range(3):
            axis_vec = affine[:3, axis_idx]
            if axis_idx == dim:
                axis_vec = axis_vec.copy()
                axis_vec /= np.sqrt(np.dot(axis_vec, axis_vec))
            axes.append(axis_vec)
        #Pull out the translation
        trans = affine[:3, 3]
        #Determine the shape / dtype of the result data array and create it
        result_shape = list(shape)
        while dim >= len(result_shape):
            result_shape.append(1)
        result_shape[dim] = n_inputs
        result_dtype = max(input_wrp.nii_img.get_data_dtype()
                           for input_wrp in seq)
        result_data = np.empty(result_shape, dtype=result_dtype)
        #Start with the header info from the first input
        hdr_info = {'qform' : first_hdr.get_qform(),
                    'qform_code' : first_hdr['qform_code'],
                    'sform' : first_hdr.get_sform(),
                    'sform_code' : first_hdr['sform_code'],
                    'dim_info' : list(first_hdr.get_dim_info()),
                    'xyzt_units' : list(first_hdr.get_xyzt_units()),
                   }
        try:
            hdr_info['slice_duration'] = first_hdr.get_slice_duration()
        except HeaderDataError:
            hdr_info['slice_duration'] = None
        try:
            hdr_info['intent'] = first_hdr.get_intent()
        except HeaderDataError:
            hdr_info['intent'] = None
        try:
            hdr_info['slice_times'] = first_hdr.get_slice_times()
        except HeaderDataError:
            hdr_info['slice_times'] = None
        #Fill the data array, check header consistency
        data_slices: List[Union[slice, int]] = [slice(None)] * len(result_shape)
        for dim_idx, dim_size in enumerate(result_shape):
            if dim_size == 1:
                data_slices[dim_idx] = 0
        last_trans = None #Keep track of the translation from last input
        for input_idx in range(n_inputs):
            input_wrp = seq[input_idx]
            input_nii = input_wrp.nii_img
            input_aff = input_nii.affine
            input_hdr = input_nii.header
            #Check that the affines match appropriately
            for axis_idx, axis_vec in enumerate(axes):
                in_vec = input_aff[:3, axis_idx]
                #If we are joining on this dimension
                if axis_idx == dim:
                    #Allow scaling difference as it will be updated later
                    in_vec = in_vec.copy()
                    in_vec /= np.sqrt(np.dot(in_vec, in_vec))
                    in_trans = input_aff[:3, 3]
                    if not last_trans is None:
                        #Must be translated along the axis
                        trans_diff = in_trans - last_trans
                        if not np.allclose(trans_diff, 0.0):
                            trans_diff /= np.sqrt(np.dot(trans_diff, trans_diff))
                        if (np.allclose(trans_diff, 0.0) or
                            not np.allclose(np.dot(trans_diff, in_vec),
                                            1.0,
                                            atol=1e-6)
                           ):
                            raise ValueError("Slices must be translated along the "
                                             "normal direction")
                    #Update reference to last translation
                    last_trans = in_trans
                #Check that axis vectors match
                if not np.allclose(in_vec, axis_vec, atol=5e-4):
                    raise ValueError("Cannot join images with different "
                                     "orientations.")
            data_slices[dim] = input_idx
            result_data[tuple(data_slices)] = np.asanyarray(input_nii.dataobj).squeeze()
            if input_idx != 0:
                if (hdr_info['qform'] is None or
                    input_hdr.get_qform() is None or
                    not np.allclose(input_hdr.get_qform(), hdr_info['qform'])
                   ):
                    hdr_info['qform'] = None
                if input_hdr['qform_code'] != hdr_info['qform_code']:
                    hdr_info['qform_code'] = None
                if (hdr_info['sform'] is None or
                    input_hdr.get_sform() is None or
                    not np.allclose(input_hdr.get_sform(), hdr_info['sform'])
                   ):
                    hdr_info['sform'] = None
                if input_hdr['sform_code'] != hdr_info['sform_code']:
                    hdr_info['sform_code'] = None
                in_dim_info = list(input_hdr.get_dim_info())
                if in_dim_info != hdr_info['dim_info']:
                    for idx in range(3):
                        if in_dim_info[idx] != hdr_info['dim_info'][idx]:
                            hdr_info['dim_info'][idx] = None
                in_xyzt_units = list(input_hdr.get_xyzt_units())
                if in_xyzt_units != hdr_info['xyzt_units']:
                    for idx in range(2):
                        if in_xyzt_units[idx] != hdr_info['xyzt_units'][idx]:
                            hdr_info['xyzt_units'][idx] = None
                try:
                    if input_hdr.get_slice_duration() != hdr_info['slice_duration']:
                        hdr_info['slice_duration'] = None
                except HeaderDataError:
                    hdr_info['slice_duration'] = None
                try:
                    if input_hdr.get_intent() != hdr_info['intent']:
                        hdr_info['intent'] = None
                except HeaderDataError:
                    hdr_info['intent'] = None
                try:
                    if input_hdr.get_slice_times() != hdr_info['slice_times']:
                        hdr_info['slice_times'] = None
                except HeaderDataError:
                    hdr_info['slice_times'] = None
        #If we joined along a spatial dim, rescale the appropriate axis
        scaled_dim_dir = None
        if dim < 3:
            scaled_dim_dir = seq[1].nii_img.affine[:3, 3] - trans
            affine[:3, dim] = scaled_dim_dir
        #Create the resulting Nifti and wrapper
        result_nii = Nifti1Image(result_data, affine)
        result_hdr = result_nii.header
        #Update the header with any info that is consistent across inputs
        if hdr_info['qform'] is not None and hdr_info['qform_code'] is not None:
            if not scaled_dim_dir is None:
                hdr_info['qform'][:3, dim] = scaled_dim_dir
            result_nii.set_qform(hdr_info['qform'],
                                 int(hdr_info['qform_code']),
                                 update_affine=True)
        if hdr_info['sform'] is not None and hdr_info['sform_code'] is not None:
            if not scaled_dim_dir is None:
                hdr_info['sform'][:3, dim] = scaled_dim_dir
            result_nii.set_sform(hdr_info['sform'],
                                 int(hdr_info['sform_code']),
                                 update_affine=True)
        if hdr_info['dim_info'] is not None:
            result_hdr.set_dim_info(*hdr_info['dim_info'])
            slice_dim = hdr_info['dim_info'][2]
        else:
            slice_dim = None
        if hdr_info['intent'] is not None:
            result_hdr.set_intent(*hdr_info['intent'])
        if hdr_info['xyzt_units'] is not None:
            result_hdr.set_xyzt_units(*hdr_info['xyzt_units'])
        if hdr_info['slice_duration'] is not None:
            result_hdr.set_slice_duration(hdr_info['slice_duration'])
        if hdr_info['slice_times'] is not None:
            result_hdr.set_slice_times(hdr_info['slice_times'])
        # Create the meta data extension and insert it
        result_ext = DcmMetaExtension.from_sequence(
            [elem.meta_ext for elem in seq],
            dim,
            affine,
            slice_dim,
        )
        if dim_key is not None:
            if dim < 3:
                warnings.warn("Ignoring 'dim_key' for slice dim")
            elif dim == 3:
                result_ext.time_dim = dim_key
            else:
                result_ext.vector_dim = dim_key
        result_hdr.extensions.append(result_ext)
        return NiftiWrapper(result_nii)