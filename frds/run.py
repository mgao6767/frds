from typing import Callable, Dict
from .professor import Professor


def main(
    measures_to_estimate=[],
    gui=False,
    progress_callback: Callable = None,
    config: Dict = None,
):
    """Main entrant of frds

    Parameters
    ----------
    measures_to_estimate : list, optional
        list of measure names to estimate, by default []
    gui : bool, optional
        if in GUI mode, by default False
    progress_callback : Callable, optional
        function used to update progress message, by default None
    """
    import inspect
    import frds.measures
    from multiprocessing import (
        get_all_start_methods,
        set_start_method,
        get_start_method,
    )

    # Use 'fork' if available
    if get_start_method(allow_none=True) is None:
        if "fork" in get_all_start_methods():
            set_start_method("fork")

    # Use standard print function if not in GUI
    progress_func = print if not gui else progress_callback.emit

    # Default to estimate all measures using default parameters
    if not gui:
        measures = [
            measure()
            for _, measure in inspect.getmembers(frds.measures, inspect.isclass)
            if not inspect.isabstract(measure)
        ]
    else:
        measures = [
            measure()
            for name, measure in inspect.getmembers(
                frds.measures, inspect.isclass
            )
            if name in measures_to_estimate
        ]

    # Professor at work!
    with Professor(config=config, progress=progress_func) as prof:
        prof.calculate(measures)


if __name__ == "__main__":

    from frds import wrds_username, wrds_password, result_dir, data_dir

    config = dict(
        wrds_username=wrds_username,
        wrds_password=wrds_password,
        result_dir=str(result_dir),
        data_dir=str(data_dir),
    )

    main(config=config)
