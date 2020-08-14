# FRDS - *for better and easier finance research*

![LICENSE](https://img.shields.io/github/license/mgao6767/frds?color=blue) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


[frds](https://github.com/mgao6767/frds/) is a Python framework for computing [a collection of major academic measures](/measures) used in the finance literature in a simple and straightforward way.

![GUI](/images/frds_gui.png)

```python linenums="1"
from frds import Professor
from frds.measures import AccountingRestatement, ROA, FirmSize

measures = [
    AccountingRestatement(years=1),
    AccountingRestatement(years=2),
    ROA(),
    FirmSize(),
]

config = dict(
        wrds_username="your_wrds_username",
        wrds_password="you_wrds_password",
        result_dir="~/frds/result/",
        data_dir="~/frds/data/",
    )

if __name__ == "__main__":
    with Professor(config=config) as prof:
        prof.calculate(measures)
```

Behind the scene, `frds` is responsible for downloading and cleaning datasets, estimating the measures and storing them orderly. As a researcher, you can and should focus on your research idea and implementation. Leave the dirty work to `frds`.