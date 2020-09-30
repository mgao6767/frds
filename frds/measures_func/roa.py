"""
# ROA
"""

from frds.measures_func import MeasureCategory, update_progress, setup

setup(
    measure_name="ROA",
    measure_type=MeasureCategory.CORPORATE_FINANCE,
    doc_url="https://frds.io/measures/roa",
    author="Mingze Gao",
    author_email="mingze.gao@sydney.edu.au",
)


def load_data():
    return 1


@update_progress()
def estimation(*args, data=load_data(), **kwargs):
    # TODO: below is only for demo
    import time
    import random

    n = 100

    for i in range(n):
        progress(int((i + 1) / n * 100))
        time.sleep(0.03)

    if random.random() > 0.8:
        raise ValueError

    return 1
