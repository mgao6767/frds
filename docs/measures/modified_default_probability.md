---
tags:
  - Banking
  - Credit Risk
---

# Modified Default Probability (Bank)

## Introduction

[Nagel and Purnanandam (2020)](https://doi.org/10.1093/rfs/hhz125) introduce the Modified Default Probability for banks. 

The rationale is simple and model beautiful. Specifically, banks' assets are contingent claims on borrowers' collateral assets, hence banks' equity and debt are contingent claims on these contingent claims. While borrowers' assets value may follow a lognormal distribution, banks' assets do not. Hence, using traditional Merton model directly on banks to gauge distance to default is not ideal.


## API

### ::: frds.measures.modified_merton.mod_merton_simulation

## Simulation Replication

The following figures are produced in by running `frds.measures.modified_merton.mod_merton_simulation.simulate()`.

### Figure 2

![figure2](https://github.com/mgao6767/frds/blob/main/docs/images/NagelPurnanandam2020/figure2_PayoffsAtDMat.png)

### Figure 3

![figure3](https://github.com/mgao6767/frds/blob/main/docs/images/NagelPurnanandam2020/figure3_mVe.png)

### Figure 4

![figure4](https://github.com/mgao6767/frds/blob/main/docs/images/NagelPurnanandam2020/figure4_mdef.png)

### Figure 5

![figure5_panel_a](https://github.com/mgao6767/frds/blob/main/docs/images/NagelPurnanandam2020/figure5_panel_a_mdefsingles.png)

![figure5_panel_b](https://github.com/mgao6767/frds/blob/main/docs/images/NagelPurnanandam2020/figure5_panel_b_mertalt.png)

## TODO

- [ ] Empirical estimation using actual bank data.

## References

* [Nagel ana Purnanandam (2020)](https://doi.org/10.1093/rfs/hhz125), Banks’ Risk Dynamics and Distance to Default, *The Review of Financial Studies*, 33(6), 2421–2467.

---

[:octicons-bug-24: Bug report](https://github.com/mgao6767/frds/issues/new?assignees=mgao6767&labels=&template=bug_report.md&title=%5BBUG%5D) | [:octicons-heart-24: Sponsor me](https://github.com/sponsors/mgao6767)