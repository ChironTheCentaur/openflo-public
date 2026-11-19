"""The events of an FCS file on the linear scale FlowJo gates them on in an
OpenFlo export (``fcs_values_as_flowjo_gates_them``).

Core module: no Tk, and it does not import ``openflo.pipeline``, which imports
this name back, so ``pipeline.fcs_values_as_flowjo_gates_them`` is this
function.
"""
from .flow_sample import FlowSample


def fcs_values_as_flowjo_gates_them(fcs_path, *, qc=False,
                                    compensation=None, set_number=None):
    """The events of `fcs_path` on the linear scale FlowJo gates them on in
    an OpenFlo export: compensated with `compensation` (``(channels,
    matrix)``, the one written for the sample) or not at all, and after QC
    when the sample had it, so they are the same events. A fresh FlowSample;
    nothing the editor holds is touched. `set_number` is the data set the
    sample is of a file holding several (None: the file as one sample)."""
    raw = FlowSample(fcs_path, set_number=set_number)
    if qc:
        raw.run_qc()
    if compensation is not None:
        raw.manual_compensate(compensation[1], list(compensation[0]))
    return raw.data
