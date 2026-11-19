"""OpenFlo's exception hierarchy: ``OpenFloError`` and the errors derived from
it (``FcsParseError``, ``CompensationError``, ``WspParseError``,
``GateError``, ``ClusteringError``).

Core module: no imports. No Tk, and it does not import ``openflo.pipeline``
(pipeline imports these back, so ``pipeline.OpenFloError`` and the rest are
the same classes).
"""

# All recoverable errors raised by the pipeline are subclasses of
# OpenFloError. Top-level callers can ``except OpenFloError`` to surface a
# user-friendly message, then fall through to ``except Exception`` for
# unexpected bugs (which deserve a full traceback).

class OpenFloError(Exception):
    """Base class for every error raised intentionally by OpenFlo."""


class FcsParseError(OpenFloError):
    """Raised when an FCS file can't be read or its metadata is malformed."""


class CompensationError(OpenFloError):
    """Raised when a compensation matrix can't be parsed or applied."""


class WspParseError(OpenFloError):
    """Raised when a FlowJo .wsp can't be parsed."""


class GateError(OpenFloError):
    """Raised when a gate definition is invalid or can't be applied."""


class ClusteringError(OpenFloError):
    """Raised when clustering (Phenograph CPU or RAPIDS GPU) fails."""
