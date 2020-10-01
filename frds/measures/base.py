import enum
import inspect
from functools import wraps


class MeasureCategory(enum.Enum):
    """Categories of measures"""

    CORPORATE_FINANCE = "Corporate Finance Measures"
    BANKING = "Banking Measures"
    MARKET_MICROSTRUCTURE = "Market Micro-structure Measures"


def update_progress():
    """Make avaiable a function `progress(pct: int)` to update the progress"""

    def variable_injector(func):
        @wraps(func)
        def decorator(*args, **kwargs):
            try:
                func_globals = func.__globals__  # Python 2.6+
            except AttributeError:
                func_globals = func.func_globals  # Earlier versions.
            saved_values = func_globals.copy()  # Shallow copy of dict.
            job_id = kwargs.get("job_id")
            _progress = kwargs.get("progress", print)
            progress = lambda pct: _progress({job_id: pct})
            func_globals.update(dict(progress=progress, job_id=job_id))
            try:
                result = func(*args, **kwargs)
            finally:
                func_globals = saved_values  # Undo changes.
            return result

        return decorator

    return variable_injector


def setup(
    measure_name: str, measure_type: MeasureCategory, doc_url: str, *args, **kwargs
):
    """Setup the estimation code"""
    # A hack to inject variables into the caller's scope
    stack = inspect.stack()
    try:
        locals_ = stack[1][0].f_locals
    finally:
        del stack
    locals_["MEASURE_NAME"] = measure_name
    locals_["MEASURE_TYPE"] = measure_type
    locals_["DOC_URL"] = doc_url
