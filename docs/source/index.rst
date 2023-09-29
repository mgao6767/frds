================================
Financial Research Data Services
================================

|frds|, *Financial Research Data Services*, is a Python library to simplify the 
complexities often encountered in financial research. It provides a collection 
of ready-to-use methods for computing a wide array of measures in the literature. 

It is developed by Dr. `Mingze Gao <http://mingze-gao.com>`_  from the University of Sydney, as 
a personal project during his postdoctoral research fellowship.

|GitHub license|

.. |GitHub license| image:: https://img.shields.io/github/license/mgao6767/frds?color=blue
   :target: https://github.com/mgao6767/frds/blob/master/LICENSE

.. |frds| replace:: :code:`frds`

.. important:: 
   This project is under active development. 
   Breaking changes may be expected.

   If there's any issue (likely), please contact me at 
   `mingze.gao@sydney.edu.au <mailto:mingze.gao@sydney.edu.au>`_.

------------
Quick start
------------

|frds| is available on `PyPI <https://pypi.org/project/frds/>`_ and can be 
installed via ``pip``.

.. code-block:: bash

   pip install frds

The structure of |frds| is simple:

* :mod:`frds.measures` provides a collection of measures.
* :mod:`frds.algorithms` provides a collection of algorithms.
* :mod:`frds.datasets` provides example datasets.
    
--------
Examples
--------

|frds| aims to provide out-of-the-box.


Measure
-------

:class:`frds.measures.DistressInsurancePremium` estimates
:doc:`/measures/distress_insurance_premium`, a systemic risk measure of a 
hypothetical insurance premium against a systemic financial distress, which is 
defined as total losses that exceed a given threshold, e.g., 15%, of total bank 
liabilities.

.. _`Huang, Zhou, and Zhu (2009)`: https://doi.org/10.1016/j.jbankfin.2009.05.017

.. |HZZ2009| replace:: `Huang, Zhou, and Zhu (2009)`_

>>> import numpy as np
>>> from frds.measures import DistressInsurancePremium
>>> # hypothetical implied default probabilities of 6 banks
>>> default_probabilities = np.array([0.02, 0.10, 0.03, 0.20, 0.50, 0.15]) 
>>> correlations = np.array(
...     [
...         [ 1.000, -0.126, -0.637, 0.174,  0.469,  0.283],
...         [-0.126,  1.000,  0.294, 0.674,  0.150,  0.053],
...         [-0.637,  0.294,  1.000, 0.073, -0.658, -0.085],
...         [ 0.174,  0.674,  0.073, 1.000,  0.248,  0.508],
...         [ 0.469,  0.150, -0.658, 0.248,  1.000, -0.370],
...         [ 0.283,  0.053, -0.085, 0.508, -0.370,  1.000],
...     ]
... )
>>> dip = DistressInsurancePremium()
>>> dip.estimate(default_probabilities, correlations)       
0.2865733550799999

Algorithm
---------

Use :class:`frds.algorithms.GARCHModel` to estimate a GARCH(1,1) model.
The results are as good as those obtained from other software or libraries.

>>> import pandas as pd
>>> from frds.algorithms import GARCHModel
>>> data_url = "https://www.stata-press.com/data/r18/stocks.dta"
>>> df = pd.read_stata(data_url, convert_dates=["date"])
>>> returns = df["nissan"].to_numpy() * 100
>>> # Start to fit the model
>>> model = GARCHModel(returns)
>>> res = model.fit() # [mu, omega, alpha, beta, log-likelihood]
>>> [f"{p:.4f}" for p in res]
['0.0193', '0.0570', '0.0905', '0.8984', '-4086.4874']

---------
Read more
---------

.. toctree::
    :maxdepth: 2

    measures/index
    algorithms/index
    datasets/index

.. toctree::
    :hidden:
    :caption: Other
    :titlesonly:

    GitHub <https://github.com/mgao6767/frds/>
    Mingze Gao <https://mingze-gao.com>
