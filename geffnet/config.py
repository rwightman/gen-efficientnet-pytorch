""" Global Config and Constants
"""

__all__ = ['is_exportable', 'is_scriptable', 'set_exportable', 'set_scriptable']

# Set to True if exporting a model with Same padding via ONNX
_EXPORTABLE = False

# Set to True if wanting to use torch.jit.script on a model
_SCRIPTABLE = False


# Set to True to force no use of torch.jit.script for model activations
_NO_JIT = False


def is_exportable():
    return _EXPORTABLE


def set_exportable(value):
    global _EXPORTABLE
    _EXPORTABLE = value


def is_scriptable():
    return _SCRIPTABLE


def set_scriptable(value):
    global _SCRIPTABLE
    _SCRIPTABLE = value


def is_no_jit():
    return _NO_JIT


def set_no_jit(value):
    global _NO_JIT
    _NO_JIT = value
