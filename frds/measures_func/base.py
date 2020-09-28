import enum
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
            progress = lambda pct: _progress((job_id, pct))
            func_globals.update(dict(progress=progress, job_id=job_id))
            try:
                result = func(*args, **kwargs)
            finally:
                func_globals = saved_values  # Undo changes.
            return result

        return decorator

    return variable_injector
