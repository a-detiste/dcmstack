"""Define extension for Nifti files that summarizes meta data from source DICOMs
"""
from __future__ import print_function

from functools import cached_property
from typing import Optional, Sequence, Tuple, Union
import sys, re, json, warnings
from copy import deepcopy
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import numpy as np
import nibabel as nb
from nibabel.nifti1 import Nifti1Extension

from .globals import SORT_GUESSES
from .utils import iteritems, unicode_str, PY2
from .info import __version__


dcm_meta_ecode = 0


_meta_version = 0.7


_req_base_keys_map= {
    0.5 : set((
        'dcmmeta_affine',
        'dcmmeta_slice_dim',
        'dcmmeta_shape',
        'dcmmeta_version',
        'global',
    )),
    0.6 : set((
        'dcmmeta_affine',
        'dcmmeta_reorient_transform',
        'dcmmeta_slice_dim',
        'dcmmeta_shape',
        'dcmmeta_version',
        'global',
    )),
    0.7 : set((
        'dcmmeta_affine',
        'dcmmeta_reorient_transform',
        'dcmmeta_slice_dim',
        'dcmmeta_time_dim',
        'dcmmeta_vector_dim',
        'dcmmeta_shape',
        'dcmmeta_version',
        'extract_version',
        'global',
    )),
}
'''Minimum required keys in the base dictionaty to be considered valid'''


def is_constant(sequence, period=None):
    '''Returns true if all elements in (each period of) the sequence are equal.

    Parameters
    ----------
    sequence : sequence
        The sequence of elements to check.

    period : int
        If not None then each subsequence of that length is checked.
    '''
    if period is None:
        return all(val == sequence[0] for val in sequence)
    else:
        if period <= 1:
            raise ValueError('The period must be greater than one')
        seq_len = len(sequence)
        if seq_len % period != 0:
            raise ValueError('The sequence length is not evenly divisible by '
                             'the period length.')
        for period_idx in range(seq_len // period):
            start_idx = period_idx * period
            end_idx = start_idx + period
            if not all(val == sequence[start_idx]
                       for val in sequence[start_idx:end_idx]):
                return False

    return True


def is_repeating(sequence, period):
    '''Returns true if the elements in the sequence repeat with the given
    period.

    Parameters
    ----------
    sequence : sequence
        The sequence of elements to check.

    period : int
        The period over which the elements should repeat.
    '''
    seq_len = len(sequence)
    if period <= 1 or period >= seq_len:
        raise ValueError('The period must be greater than one and less than '
                         'the length of the sequence')
    if seq_len % period != 0:
        raise ValueError('The sequence length is not evenly divisible by the '
                         'period length.')

    for period_idx in range(1, seq_len // period):
        start_idx = period_idx * period
        end_idx = start_idx + period
        if sequence[start_idx:end_idx] != sequence[:period]:
            return False

    return True


class InvalidExtensionError(Exception):
    def __init__(self, msg):
        '''Exception denoting than a DcmMetaExtension is invalid.'''
        self.msg = msg

    def __str__(self):
        return 'The extension is not valid: %s' % self.msg


class DcmMetaExtension(Nifti1Extension):
    '''Nifti extension for storing a summary of the meta data from the source
    DICOM files.
    '''

    @classmethod
    def make_empty(
        klass, 
        shape: Tuple[int, ...], 
        affine: np.ndarray, 
        reorient_transform: Optional[np.ndarray] = None,
        slice_dim: Optional[int] = None,
        time_dim: Optional[str] = None,
        vector_dim: Optional[str] = None,
        extract_version: Optional[str] = None,
    ) -> "DcmMetaExtension":
        '''Make an empty DcmMetaExtension

        Parameters
        ----------
        shape
            The shape of the data associated with this extension.

        affine
            The RAS affine for the data associated with this extension.

        reorient_transform
            The transformation matrix representing any reorientation of the
            data array.

        slice_dim
            The index of the slice dimension for the data associated with this
            extension

        time_dim
            The attribute used to order the time dimension

        vector_dim
            The attribute used to order the vector dimension
        '''
        result = klass(dcm_meta_ecode, '{}')
        result._content['global'] = {}
        result._content['global']['const'] = {}
        result._content['global']['slices'] = {}
        if len(shape) > 3 and shape[3] != 1:
            result._content['time'] = {}
            result._content['time']['samples'] = {}
            result._content['time']['slices'] = {}
        else:
            time_dim = None
        if len(shape) > 4:
            result._content['vector'] = {}
            result._content['vector']['samples'] = {}
            result._content['vector']['slices'] = {}
        else:
            vector_dim = None
        result._content['dcmmeta_shape'] = []
        result.shape = shape
        result.affine = affine
        result.reorient_transform = reorient_transform
        result.slice_dim = slice_dim
        result.time_dim = time_dim
        result.vector_dim = vector_dim
        result.version = _meta_version
        result.extract_version = __version__ if extract_version is None else extract_version
        return result

    @classmethod
    def from_sequence(
        klass, 
        seq: Sequence["DcmMetaExtension"], 
        dim: int, 
        affine: Optional[np.ndarray] = None, 
        slice_dim: Optional[int] = None,
        time_dim: Optional[str] = None,
        vector_dim: Optional[str] = None,
    ) -> "DcmMetaExtension":
        '''Concatenate sequence `seq` of extensions along dimension `dim` 

        Parameters
        ----------
        seq
            The sequence of DcmMetaExtension objects to merge

        dim
            The dimension to merge the extensions along.

        affine
            The affine to use in the resulting extension. If None, the affine
            from the first extension in `seq` will be used.

        slice_dim
            The slice dimension to use in the resulting extension. If None, the
            slice dimension from the first extension in `seq` will be used.

        time_dim
            The attribute used to order the time dimension

        vector_dim
            The attribute used to order the vector dimension
        '''
        if not 0 <= dim < 5:
            raise ValueError("The argument 'dim' must be in the range [0, 5).")
        # Determine the output shape and create empty extension
        n_inputs = len(seq)
        first_input = seq[0]
        input_shape = first_input.shape
        if len(input_shape) > dim and input_shape[dim] != 1:
            raise ValueError("The dim must be singular or not exist for the "
                             "inputs.")
        output_shape = list(input_shape)
        while len(output_shape) <= dim:
            output_shape.append(1)
        output_shape[dim] = n_inputs
        output_shape = tuple(output_shape)
        if affine is None:
            affine = first_input.affine
        if slice_dim is None:
            slice_dim = first_input.slice_dim
        if time_dim is None:
            time_dim = first_input.time_dim
        if vector_dim is None:
            vector_dim = first_input.vector_dim
        result = klass.make_empty(
            output_shape,
            affine,
            None,
            slice_dim,
            time_dim,
            vector_dim,
            first_input.extract_version,
        )
        # Need to initialize the result with the first extension in 'seq'
        result_slc_norm = result.slice_normal
        first_slc_norm = first_input.slice_normal
        use_slices = (not result_slc_norm is None and
                      not first_slc_norm is None and
                      np.allclose(result_slc_norm, first_slc_norm))
        for classes in first_input.valid_classes:
            if classes[1] == 'slices' and not use_slices:
                continue
            result._content[classes[0]][classes[1]] = \
                deepcopy(first_input.get_class_dict(classes))
        #Adjust the shape to what the extension actually contains
        shape = list(result.shape)
        shape[dim] = 1
        result.shape = shape
        #Initialize reorient transform
        reorient_transform = first_input.reorient_transform
        #Add the other extensions, updating the shape as we go
        for input_ext in seq[1:]:
            #If the affines or reorient_transforms don't match, we set the
            #reorient_transform to None as we can not reliably use it to update
            #directional meta data
            if ((reorient_transform is None or
                 input_ext.reorient_transform is None) or
                not (np.allclose(input_ext.affine, affine) or
                     np.allclose(input_ext.reorient_transform,
                                 reorient_transform)
                    )
               ):
                reorient_transform = None
            result._insert(dim, input_ext)
            shape[dim] += 1
            result.shape = shape
        #Set the reorient transform
        result.reorient_transform = reorient_transform
        #Try simplifying any keys in global slices
        for key in list(result.get_class_dict(('global', 'slices'))):
            result._simplify(key)
        return result

    def guess_extra_dims(self) -> None:
        extra_shape = self.shape[3:]
        if extra_shape:
            if extra_shape[0] != 1 and self.time_dim is None:
                meta = self.get_class_dict(("time", "samples"))
                meta = {k : v for k, v in meta.items() if len(v) == len(set(v))}
                for key in SORT_GUESSES:
                    if key in meta:
                        self.time_dim = key
                        break
                else:
                    meta = self.get_class_dict(("global", "slices"))
                    meta = {k : v for k, v in meta.items() if len(v) == len(set(v))}
                    for key in SORT_GUESSES:
                        if key in meta:
                            self.time_dim = key
                            break
            if len(extra_shape) == 2 and extra_shape[1] != 1 and self.vector_dim is None:
                pass # TODO

    @property
    def shape(self):
        '''The shape of the data associated with the meta data. Defines the
        number of values for the meta data classifications.'''
        return tuple(self._content['dcmmeta_shape'])

    @shape.setter
    def shape(self, value):
        if not (3 <= len(value) < 6):
            raise ValueError("The shape must have a length between three and "
                             "six")
        # TODO: If num slice / time / vector change we should remove corresponding
        #       meta data
        self._content['dcmmeta_shape'][:] = value

    @property
    def affine(self):
        '''The affine associated with the meta data. If this differs from the
        image affine, the per-slice meta data will not be used. '''
        return np.array(self._content['dcmmeta_affine'])

    @affine.setter
    def affine(self, value):
        if value.shape != (4, 4):
            raise ValueError("Invalid shape for affine")
        self._content['dcmmeta_affine'] = value.tolist()

    @property
    def reorient_transform(self):
        '''The transformation due to reorientation of the data array. Can be
        used to update directional DICOM meta data (after converting to RAS if
        needed) into the same space as the affine.'''
        if self.version < 0.6:
            return None
        if self._content['dcmmeta_reorient_transform'] is None:
            return None
        return np.array(self._content['dcmmeta_reorient_transform'])

    @reorient_transform.setter
    def reorient_transform(self, value):
        if not value is None and value.shape != (4, 4):
            raise ValueError("The reorient_transform must be none or (4,4) "
            "array")
        if value is None:
            self._content['dcmmeta_reorient_transform'] = None
        else:
            self._content['dcmmeta_reorient_transform'] = value.tolist()

    @property
    def slice_dim(self) -> Optional[int]:
        '''The index of the slice dimension associated with the per-slice meta data.'''
        return self._content['dcmmeta_slice_dim']

    @slice_dim.setter
    def slice_dim(self, value: Optional[int]):
        if value is not None and not (0 <= value < 3):
            raise ValueError("The slice dimension must be between zero and "
                             "two")
        self._content['dcmmeta_slice_dim'] = value
    
    @property
    def time_dim(self) -> Optional[str]:
        '''The meta data key used to order the 'time' dimension'''
        return self._content.get('dcmmeta_time_dim')
    
    @time_dim.setter
    def time_dim(self, value: Optional[str]) -> None:
        if value is not None:
            # TODO: Need better validation here, maybe wait for metasum work...
            if value in self.get_class_dict(("global", "const")) or value not in self.get_keys():
                raise ValueError("Invalid attribute for indexing: {value}")
        self._content['dcmmeta_time_dim'] = value

    @property
    def vector_dim(self) -> Optional[str]:
        '''The meta data key used to order the 'vector' dimension'''
        return self._content.get('dcmmeta_vector_dim')

    @vector_dim.setter
    def vector_dim(self, value: Optional[str]) -> None:
        if value is not None:
            # TODO: Need better validation here, maybe wait for metasum work...
            if value in self.get_class_dict(("global", "const")) or value not in self.get_keys():
                raise ValueError("Invalid attribute for indexing: {value}")
        self._content['dcmmeta_vector_dim'] = value

    @property
    def version(self):
        '''The version of the meta data extension.'''
        return self._content['dcmmeta_version']

    @version.setter
    def version(self, value):
        '''Set the version of the meta data extension.'''
        self._content['dcmmeta_version'] = value

    @property
    def extract_version(self) -> Optional[str]:
        return self._content["extract_version"]
    
    @extract_version.setter
    def extract_version(self, value: Optional[str]) -> None:
        self._content["extract_version"] = value

    @property
    def slice_normal(self):
        '''The slice normal associated with the per-slice meta data.'''
        slice_dim = self.slice_dim
        if slice_dim is None:
            return None
        return np.array(self.affine[slice_dim][:3])

    @property
    def n_slices(self):
        '''The number of slices associated with the per-slice meta data.'''
        slice_dim = self.slice_dim
        if slice_dim is None:
            return None
        return self.shape[slice_dim]

    # TODO: Can we provide better aliases for these, like:
    # global, per_time, per_time_slice, per_vec, per_vec_slice, per_slice
    classifications = (('global', 'const'), 
                       ('global', 'slices'),
                       ('time', 'samples'),
                       ('time', 'slices'),
                       ('vector', 'samples'),
                       ('vector', 'slices'),
                      )
    '''The classifications used to separate meta data based on if and how the
    values repeat. Each class is a tuple with a base class and a sub class.'''

    def get_valid_classes(self):
        '''Return the meta data classifications that are valid for this
        extension.

        Returns
        -------
        valid_classes : tuple
            The classifications that are valid for this extension (based on its
            shape).

        '''
        shape = self.shape
        n_dims = len(shape)
        if n_dims == 3:
            return self.classifications[:2]
        elif n_dims == 4:
            return self.classifications[:4]
        elif n_dims == 5:
            if shape[3] != 1:
                return self.classifications
            else:
                return self.classifications[:2] + self.classifications[-2:]
        else:
            raise ValueError("There must be 3 to 5 dimensions.")
        
    valid_classes = cached_property(get_valid_classes)

    def get_multiplicity(self, classification):
        '''Get the number of meta data values for all meta data of the provided
        classification.

        Parameters
        ----------
        classification : tuple
            The meta data classification.

        Returns
        -------
        multiplicity : int
            The number of values for any meta data of the provided
            `classification`.
        '''
        if not classification in self.valid_classes:
            raise ValueError("Invalid classification: %s" % classification)

        base, sub = classification
        shape = self.shape
        n_vals = 1
        if sub == 'slices':
            n_vals = self.n_slices
            if n_vals is None:
                return 0
            if base == 'vector':
                n_vals *= shape[3]
            elif base == 'global':
                for dim_size in shape[3:]:
                    n_vals *= dim_size
        elif sub == 'samples':
            if base == 'time':
                n_vals = shape[3]
                if len(shape) == 5:
                    n_vals *= shape[4]
            elif base == 'vector':
                n_vals = shape[4]

        return n_vals

    def check_valid(self):
        '''Check if the extension is valid.

        Raises
        ------
        InvalidExtensionError
            The extension is missing required meta data or classifications, or
            some element(s) have the wrong number of values for their
            classification.
        '''
        #Check for the required base keys in the json data
        if not _req_base_keys_map[self.version] <= set(self._content):
            raise InvalidExtensionError('Missing one or more required keys')

        #Check the orientation/shape/version
        if self.affine.shape != (4, 4):
            raise InvalidExtensionError('Affine has incorrect shape')
        slice_dim = self.slice_dim
        if slice_dim is not None:
            if not (0 <= slice_dim < 3):
                raise InvalidExtensionError('Slice dimension is not valid')
        if not (3 <= len(self.shape) < 6):
            raise InvalidExtensionError('Shape is not valid')

        #Check all required meta dictionaries, make sure values have correct
        #multiplicity
        valid_classes = self.valid_classes
        for classes in valid_classes:
            if not classes[0] in self._content:
                raise InvalidExtensionError('Missing required base '
                                            'classification %s' % classes[0])
            if not classes[1] in self._content[classes[0]]:
                raise InvalidExtensionError(('Missing required sub '
                                             'classification %s in base '
                                             'classification %s') % classes)
            cls_meta = self.get_class_dict(classes)
            cls_mult = self.get_multiplicity(classes)
            if cls_mult == 0 and len(cls_meta) != 0:
                raise InvalidExtensionError('Slice dim is None but per-slice '
                                            'meta data is present')
            elif cls_mult > 1:
                for key, vals in iteritems(cls_meta):
                    n_vals = len(vals)
                    if n_vals != cls_mult:
                        msg = (('Incorrect number of values for key %s with '
                                'classification %s, expected %d found %d') %
                               (key, classes, cls_mult, n_vals)
                              )
                        raise InvalidExtensionError(msg)

        #Check that all keys are uniquely classified
        for classes in valid_classes:
            for other_classes in valid_classes:
                if classes == other_classes:
                    continue
                intersect = (set(self.get_class_dict(classes)) &
                             set(self.get_class_dict(other_classes))
                            )
                if len(intersect) != 0:
                    raise InvalidExtensionError("One or more keys have "
                                                "multiple classifications")

    def get_keys(self):
        '''Get a list of all the meta data keys that are available.'''
        keys = []
        for base_class, sub_class in self.valid_classes:
            keys += self._content[base_class][sub_class].keys()
        return keys

    def get_classification(self, key):
        '''Get the classification for the given `key`.

        Parameters
        ----------
        key : str
            The meta data key.

        Returns
        -------
        classification : tuple or None
            The classification tuple for the provided key or None if the key is
            not found.

        '''
        for base_class, sub_class in self.valid_classes:
            if key in self._content[base_class][sub_class]:
                    return (base_class, sub_class)

        return None

    def get_class_dict(self, classification):
        '''Get the dictionary for the given classification.

        Parameters
        ----------
        classification : tuple
            The meta data classification.

        Returns
        -------
        meta_dict : dict
            The dictionary for the provided classification.
        '''
        base, sub = classification
        return self._content[base][sub]

    def get_values(self, key):
        '''Get all values for the provided key.

        Parameters
        ----------
        key : str
            The meta data key.

        Returns
        -------
        values
             The value or values for the given key. The number of values
             returned depends on the classification (see 'get_multiplicity').
        '''
        classification = self.get_classification(key)
        if classification is None:
            return None
        return self.get_class_dict(classification)[key]

    def get_values_and_class(self, key):
        '''Get the values and the classification for the provided key.

        Parameters
        ----------
        key : str
            The meta data key.

        Returns
        -------
        vals_and_class : tuple
            None for both the value and classification if the key is not found.

        '''
        classification = self.get_classification(key)
        if classification is None:
            return (None, None)
        return (self.get_class_dict(classification)[key], classification)

    def filter_meta(self, filter_func):
        '''Filter the meta data.

        Parameters
        ----------
        filter_func : callable
            Must take a key and values as parameters and return True if they
            should be filtered out.

        '''
        for classes in self.valid_classes:
            filtered = []
            curr_dict = self.get_class_dict(classes)
            for key, values in iteritems(curr_dict):
                if filter_func(key, values):
                    filtered.append(key)
            for key in filtered:
                del curr_dict[key]

    def clear_slice_meta(self):
        '''Clear all meta data that is per slice.'''
        for base_class, sub_class in self.valid_classes:
            if sub_class == 'slices':
                self.get_class_dict((base_class, sub_class)).clear()

    def get_subset(self, dim, idx):
        '''Get a DcmMetaExtension containing a subset of the meta data.

        Parameters
        ----------
        dim : int
            The dimension we are taking the subset along.

        idx : int
            The position on the dimension `dim` for the subset.

        Returns
        -------
        result : DcmMetaExtension
            A new DcmMetaExtension corresponding to the subset.

        '''
        if not 0 <= dim < 5:
            raise ValueError("The argument 'dim' must be in the range [0, 5).")
        shape = self.shape
        valid_classes = self.valid_classes
        #Make an empty extension for the result
        result_shape = list(shape)
        result_shape[dim] = 1
        while result_shape[-1] == 1 and len(result_shape) > 3:
            result_shape = result_shape[:-1]
        result_shape = tuple(result_shape)
        result = self.make_empty(result_shape,
                                 self.affine,
                                 self.reorient_transform,
                                 self.slice_dim,
                                 self.time_dim,
                                 self.vector_dim,
                                )
        for src_class in valid_classes:
            #Constants remain constant
            if src_class == ('global', 'const'):
                for key, val in iteritems(self.get_class_dict(src_class)):
                    result.get_class_dict(src_class)[key] = deepcopy(val)
                continue
            if dim == self.slice_dim:
                if src_class[1] != 'slices':
                    for key, vals in iteritems(self.get_class_dict(src_class)):
                        result.get_class_dict(src_class)[key] = deepcopy(vals)
                else:
                    result._copy_slice(self, src_class, idx)
            elif dim < 3:
                for key, vals in iteritems(self.get_class_dict(src_class)):
                    result.get_class_dict(src_class)[key] = deepcopy(vals)
            elif dim == 3:
                result._copy_sample(self, src_class, 'time', idx)
            else:
                result._copy_sample(self, src_class, 'vector', idx)
        return result

    def to_json(self):
        '''Return the extension encoded as a JSON string.'''
        self.check_valid()
        return json.dumps(self._content, indent=4)

    @classmethod
    def from_json(klass, json_str):
        '''Create an extension from the JSON string representation.'''
        result = klass(dcm_meta_ecode, json_str)
        result.check_valid()
        return result

    @classmethod
    def from_runtime_repr(klass, runtime_repr):
        '''Create an extension from the Python runtime representation (nested
        dictionaries).
        '''
        result = klass(dcm_meta_ecode, '{}')
        result._content = runtime_repr
        result.check_valid()
        return result

    def __str__(self) -> str:
        return json.dumps(self._content, indent=4)

    def __eq__(self, other):
        if not np.allclose(self.affine, other.affine):
            return False
        if self.shape != other.shape:
            return False
        if self.slice_dim != other.slice_dim:
            return False
        if self.version != other.version:
            return False
        for classes in self.valid_classes:
            if (dict(self.get_class_dict(classes)) !=
               dict(other.get_class_dict(classes))):
                return False

        return True

    def _unmangle(self, value):
        '''Go from extension data to runtime representation.'''
        if not isinstance(value, unicode_str):
            value = value.decode('utf-8')
        #Its not possible to preserve order while loading with python 2.6
        kwargs = {}
        if sys.version_info >= (2, 7):
            kwargs['object_pairs_hook'] = OrderedDict
        return json.loads(value, **kwargs)

    def _mangle(self, value):
        '''Go from runtime representation to extension data.'''
        res = json.dumps(value, indent=4)
        # Python 2 leaves some trailing white-space in the JSON output while
        # python 3 does not. We strip it so output is binary identical across 
        # versions
        if PY2:
            res = re.sub('[ \t]+$', '', res, 0, re.M)
        return res.encode('utf-8')

    _const_tests = {('global', 'slices') : (('global', 'const'),
                                            ('vector', 'samples'),
                                            ('time', 'samples')
                                           ),
                    ('vector', 'slices') : (('global', 'const'),
                                            ('time', 'samples')
                                           ),
                    ('time', 'slices') : (('global', 'const'),
                                         ),
                    ('time', 'samples') : (('global', 'const'),
                                           ('vector', 'samples'),
                                          ),
                    ('vector', 'samples') : (('global', 'const'),)
                   }
    '''Classification mapping showing possible reductions in multiplicity for
    values that are constant with some period.'''

    def _get_const_period(self, src_cls, dest_cls):
        '''Get the period over which we test for const-ness with for the
        given classification change.'''
        if dest_cls == ('global', 'const'):
            return None
        elif src_cls == ('global', 'slices'):
            return int(self.get_multiplicity(src_cls) // self.get_multiplicity(dest_cls))
        elif src_cls == ('vector', 'slices'): #implies dest_cls == ('time', 'samples'):
            return  self.n_slices
        elif src_cls == ('time', 'samples'): #implies dest_cls == ('vector', 'samples')
            return self.shape[3]
        assert False #Should take one of the above branches

    _repeat_tests = {('global', 'slices') : (('time', 'slices'),
                                             ('vector', 'slices')
                                            ),
                     ('vector', 'slices') : (('time', 'slices'),),
                    }
    '''Classification mapping showing possible reductions in multiplicity for
    values that are repeating with some period.'''

    def _simplify(self, key):
        '''Try to simplify (reduce the multiplicity) of a single meta data
        element by changing its classification. Return True if the
        classification is changed, otherwise False.

        Looks for values that are constant or repeating with some pattern.
        Constant elements with a value of None will be deleted.
        '''
        values, curr_class = self.get_values_and_class(key)

        #If the class is global const then just delete it if the value is None
        if curr_class == ('global', 'const'):
            if values is None:
                del self.get_class_dict(curr_class)[key]
                return True
            return False

        #Test if the values are constant with some period
        dests = self._const_tests[curr_class]
        for dest_cls in dests:
            if dest_cls[0] in self._content:
                period = self._get_const_period(curr_class, dest_cls)
                #If the period is one, the two classifications have the
                #same multiplicity so we are dealing with a degenerate
                #case (i.e. single slice data). Just change the
                #classification to the "simpler" one in this case
                if period == 1 or is_constant(values, period):
                    if period is None:
                        self.get_class_dict(dest_cls)[key] = \
                            values[0]
                    else:
                        self.get_class_dict(dest_cls)[key] = \
                            values[::period]
                    break
        else: #Otherwise test if values are repeating with some period
            if curr_class in self._repeat_tests:
                for dest_cls in self._repeat_tests[curr_class]:
                    if dest_cls[0] in self._content:
                        dest_mult = self.get_multiplicity(dest_cls)
                        if is_repeating(values, dest_mult):
                            self.get_class_dict(dest_cls)[key] = \
                                values[:dest_mult]
                            break
                else: #Can't simplify
                    return False
            else:
                return False

        del self.get_class_dict(curr_class)[key]
        return True

    _preserving_changes = {None : (('global', 'const'),
                                   ('vector', 'samples'),
                                   ('time', 'samples'),
                                   ('time', 'slices'),
                                   ('vector', 'slices'),
                                   ('global', 'slices'),
                                  ),
                           ('global', 'const') : (('vector', 'samples'),
                                                  ('time', 'samples'),
                                                  ('time', 'slices'),
                                                  ('vector', 'slices'),
                                                  ('global', 'slices'),
                                                 ),
                           ('vector', 'samples') : (('time', 'samples'),
                                                    ('global', 'slices'),
                                                   ),
                           ('time', 'samples') : (('global', 'slices'),
                                                 ),
                           ('time', 'slices') : (('vector', 'slices'),
                                                 ('global', 'slices'),
                                                ),
                           ('vector', 'slices') : (('global', 'slices'),
                                                  ),
                           ('global', 'slices') : tuple(),
                          }
    '''Classification mapping showing allowed changes when increasing the
    multiplicity.'''

    def _get_changed_class(self, key, new_class, slice_dim=None):
        '''Get an array of values corresponding to a single meta data
        element with its classification changed by increasing its
        multiplicity. This will preserve all the meta data and allow easier
        merging of values with different classifications.'''
        values, curr_class = self.get_values_and_class(key)
        if curr_class == new_class:
            return values

        if not new_class in self._preserving_changes[curr_class]:
            raise ValueError("Classification change would lose data.")

        if curr_class is None:
            curr_mult = 1
            per_slice = False
        else:
            curr_mult = self.get_multiplicity(curr_class)
            per_slice = curr_class[1] == 'slices'
        if new_class in self.valid_classes:
            new_mult = self.get_multiplicity(new_class)
            #Only way we get 0 for mult is if slice dim is undefined
            if new_mult == 0:
                new_mult = self.shape[slice_dim]
        else:
            new_mult = 1
        mult_fact = int(new_mult // curr_mult)
        if curr_mult == 1:
            values = [values]


        if per_slice:
            result = values * mult_fact
        else:
            result = []
            for value in values:
                result.extend([deepcopy(value)] * mult_fact)

        if new_class == ('global', 'const'):
            result = result[0]

        return result

    def _change_class(self, key, new_class):
        '''Change the classification of the meta data element in place. See
        _get_changed_class.'''
        values, curr_class = self.get_values_and_class(key)
        if curr_class == new_class:
            return

        self.get_class_dict(new_class)[key] = self._get_changed_class(key,
                                                                      new_class)

        if not curr_class is None:
            del self.get_class_dict(curr_class)[key]

    def _copy_slice(self, other, src_class, idx):
        '''Get a copy of the meta data from the 'other' instance with
        classification 'src_class', corresponding to the slice with index
        'idx'.'''
        if src_class[0] == 'global':
            for classes in (('time', 'samples'),
                            ('vector', 'samples'),
                            ('global', 'const')):
                if classes in self.valid_classes:
                    dest_class = classes
                    break
        elif src_class[0] == 'vector':
            for classes in (('time', 'samples'),
                            ('global', 'const')):
                if classes in self.valid_classes:
                    dest_class = classes
                    break
        else:
            dest_class = ('global', 'const')

        src_dict = other.get_class_dict(src_class)
        dest_dict = self.get_class_dict(dest_class)
        dest_mult = self.get_multiplicity(dest_class)
        stride = other.n_slices
        for key, vals in iteritems(src_dict):
            subset_vals = vals[idx::stride]

            if len(subset_vals) < dest_mult:
                full_vals = []
                for val_idx in range(dest_mult // len(subset_vals)):
                    full_vals += deepcopy(subset_vals)
                subset_vals = full_vals
            if len(subset_vals) == 1:
                subset_vals = subset_vals[0]
            dest_dict[key] = deepcopy(subset_vals)
            self._simplify(key)

    def _global_slice_subset(self, key, sample_base, idx):
        '''Get a subset of the meta data values with the classificaion
        ('global', 'slices') corresponding to a single sample along the
        time or vector dimension (as specified by 'sample_base' and 'idx').
        '''
        n_slices = self.n_slices
        shape = self.shape
        src_dict = self.get_class_dict(('global', 'slices'))
        if sample_base == 'vector':
            slices_per_vec = n_slices * shape[3]
            start_idx = idx * slices_per_vec
            end_idx = start_idx + slices_per_vec
            return src_dict[key][start_idx:end_idx]
        else:
            if not ('vector', 'samples') in self.valid_classes:
                start_idx = idx * n_slices
                end_idx = start_idx + n_slices
                return src_dict[key][start_idx:end_idx]
            else:
                result = []
                slices_per_vec = n_slices * shape[3]
                for vec_idx in range(shape[4]):
                    start_idx = (vec_idx * slices_per_vec) + (idx * n_slices)
                    end_idx = start_idx + n_slices
                    result.extend(src_dict[key][start_idx:end_idx])
                return result

    def _copy_sample(self, other, src_class, sample_base, idx):
        '''Get a copy of meta data from 'other' instance with classification
        'src_class', corresponding to one sample along the time or vector
        dimension.'''
        assert src_class != ('global', 'const')
        src_dict = other.get_class_dict(src_class)
        if src_class[1] == 'samples':
            #If we are indexing on the same dim as the src_class we need to
            #change the classification
            if src_class[0] == sample_base:
                #Time samples may become vector samples, otherwise const
                best_dest = None
                for dest_cls in (('vector', 'samples'),
                                 ('global', 'const')):
                    if (dest_cls != src_class and
                        dest_cls in self.valid_classes
                       ):
                        best_dest = dest_cls
                        break

                dest_mult = self.get_multiplicity(dest_cls)
                if dest_mult == 1:
                    for key, vals in iteritems(src_dict):
                        self.get_class_dict(dest_cls)[key] = \
                            deepcopy(vals[idx])
                else: #We must be doing time samples -> vector samples
                    stride = other.shape[3]
                    for key, vals in iteritems(src_dict):
                        self.get_class_dict(dest_cls)[key] = \
                            deepcopy(vals[idx::stride])
                    for key in src_dict.keys():
                        self._simplify(key)

            else: #Otherwise classification does not change
                #The multiplicity will change for time samples if splitting
                #vector dimension
                if src_class == ('time', 'samples'):
                    dest_mult = self.get_multiplicity(src_class)
                    start_idx = idx * dest_mult
                    end_idx = start_idx + dest_mult
                    for key, vals in iteritems(src_dict):
                        self.get_class_dict(src_class)[key] = \
                            deepcopy(vals[start_idx:end_idx])
                        self._simplify(key)
                else: #Otherwise multiplicity is unchanged
                    for key, vals in iteritems(src_dict):
                        self.get_class_dict(src_class)[key] = deepcopy(vals)
        else: #The src_class is per slice
            if src_class[0] == sample_base:
                best_dest = None
                for dest_class in self._preserving_changes[src_class]:
                    if dest_class in self.valid_classes:
                        best_dest = dest_class
                        break
                for key, vals in iteritems(src_dict):
                    self.get_class_dict(best_dest)[key] = deepcopy(vals)
            elif src_class[0] != 'global':
                if sample_base == 'time':
                    #Take a subset of vector slices
                    n_slices = self.n_slices
                    start_idx = idx * n_slices
                    end_idx = start_idx + n_slices
                    for key, vals in iteritems(src_dict):
                        self.get_class_dict(src_class)[key] = \
                            deepcopy(vals[start_idx:end_idx])
                        self._simplify(key)
                else:
                    #Time slices are unchanged
                    for key, vals in iteritems(src_dict):
                        self.get_class_dict(src_class)[key] = deepcopy(vals)
            else:
                #Take a subset of global slices
                for key, vals in iteritems(src_dict):
                    subset_vals = \
                        other._global_slice_subset(key, sample_base, idx)
                    self.get_class_dict(src_class)[key] = deepcopy(subset_vals)
                    self._simplify(key)

    def _insert(self, dim, other):
        self_slc_norm = self.slice_normal
        other_slc_norm = other.slice_normal

        #If we are not using slice meta data, temporarily remove it from the
        #other dcmmeta object
        use_slices = (not self_slc_norm is None and
                      not other_slc_norm is None and
                      np.allclose(self_slc_norm, other_slc_norm))
        other_slc_meta = {}
        if not use_slices:
            for classes in other.valid_classes:
                if classes[1] == 'slices':
                    other_slc_meta[classes] = other.get_class_dict(classes)
                    other._content[classes[0]][classes[1]] = {}
        missing_keys = list(set(self.get_keys()) - set(other.get_keys()))
        for other_classes in other.valid_classes:
            other_keys = list(other.get_class_dict(other_classes).keys())

            #Treat missing keys as if they were in global const and have a value
            #of None
            if other_classes == ('global', 'const'):
                other_keys += missing_keys

            #When possible, reclassify our meta data so it matches the other
            #classification
            for key in other_keys:
                local_classes = self.get_classification(key)
                if local_classes != other_classes:
                    local_allow = self._preserving_changes[local_classes]
                    other_allow = self._preserving_changes[other_classes]

                    if other_classes in local_allow:
                        self._change_class(key, other_classes)
                    elif not local_classes in other_allow:
                        best_dest = None
                        for dest_class in local_allow:
                            if (dest_class[0] in self._content and
                               dest_class in other_allow):
                                best_dest = dest_class
                                break
                        self._change_class(key, best_dest)

            #Insert new meta data and further reclassify as necessary
            for key in other_keys:
                if dim == self.slice_dim:
                    self._insert_slice(key, other)
                elif dim < 3:
                    self._insert_non_slice(key, other)
                elif dim == 3:
                    self._insert_sample(key, other, 'time')
                elif dim == 4:
                    self._insert_sample(key, other, 'vector')

        #Restore per slice meta if needed
        if not use_slices:
            for classes in other.valid_classes:
                if classes[1] == 'slices':
                    other._content[classes[0]][classes[1]] = \
                        other_slc_meta[classes]

    def _insert_slice(self, key, other):
        local_vals, classes = self.get_values_and_class(key)
        other_vals = other._get_changed_class(key, classes, self.slice_dim)


        #Handle some common / simple insertions with special cases
        if classes == ('global', 'const'):
            if local_vals != other_vals:
                for dest_base in ('time', 'vector', 'global'):
                    if dest_base in self._content:
                        self._change_class(key, (dest_base, 'slices'))
                        other_vals = other._get_changed_class(key,
                                                              (dest_base,
                                                               'slices'),
                                                               self.slice_dim
                                                             )
                        self.get_values(key).extend(other_vals)
                        break
        elif classes == ('time', 'slices'):
            local_vals.extend(other_vals)
        else:
            #Default to putting in global slices and simplifying later
            if classes != ('global', 'slices'):
                self._change_class(key, ('global', 'slices'))
                local_vals = self.get_class_dict(('global', 'slices'))[key]
                other_vals = other._get_changed_class(key,
                                                      ('global', 'slices'),
                                                      self.slice_dim)

            #Need to interleave slices from different volumes
            n_slices = self.n_slices
            other_n_slices = other.n_slices
            shape = self.shape
            n_vols = 1
            for dim_size in shape[3:]:
                n_vols *= dim_size

            intlv = []
            loc_start = 0
            oth_start = 0
            for vol_idx in range(n_vols):
                intlv += local_vals[loc_start:loc_start + n_slices]
                intlv += other_vals[oth_start:oth_start + other_n_slices]
                loc_start += n_slices
                oth_start += other_n_slices

            self.get_class_dict(('global', 'slices'))[key] = intlv

    def _insert_non_slice(self, key, other):
        local_vals, classes = self.get_values_and_class(key)
        other_vals = other._get_changed_class(key, classes, self.slice_dim)

        if local_vals != other_vals:
            del self.get_class_dict(classes)[key]

    def _insert_sample(self, key, other, sample_base):
        local_vals, classes = self.get_values_and_class(key)
        other_vals = other._get_changed_class(key, classes, self.slice_dim)

        if classes == ('global', 'const'):
            if local_vals != other_vals:
                self._change_class(key, (sample_base, 'samples'))
                local_vals = self.get_values(key)
                other_vals = other._get_changed_class(key,
                                                      (sample_base, 'samples'),
                                                      self.slice_dim
                                                     )
                local_vals.extend(other_vals)
        elif classes == (sample_base, 'samples'):
            local_vals.extend(other_vals)
        else:
            if classes != ('global', 'slices'):
                self._change_class(key, ('global', 'slices'))
                local_vals = self.get_values(key)
                other_vals = other._get_changed_class(key,
                                                      ('global', 'slices'),
                                                      self.slice_dim)

            shape = self.shape
            n_dims = len(shape)
            if sample_base == 'time' and n_dims == 5:
                #Need to interleave values from the time points in each vector
                #component
                n_slices = self.n_slices
                slices_per_vec = n_slices * shape[3]
                oth_slc_per_vec = n_slices * other.shape[3]

                intlv = []
                loc_start = 0
                oth_start = 0
                for vec_idx in range(shape[4]):
                    intlv += local_vals[loc_start:loc_start+slices_per_vec]
                    intlv += other_vals[oth_start:oth_start+oth_slc_per_vec]
                    loc_start += slices_per_vec
                    oth_start += oth_slc_per_vec

                self.get_class_dict(('global', 'slices'))[key] = intlv
            else:
                local_vals.extend(other_vals)

#Add our extension to nibabel
nb.nifti1.extension_codes.add_codes(((dcm_meta_ecode,
                                      "dcmmeta",
                                      DcmMetaExtension),)
                                   )
