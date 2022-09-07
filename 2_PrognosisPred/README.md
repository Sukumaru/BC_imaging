## Description

This part refers to the:

*Prognosis prediction using single-cell phenotype features* 

part of the project



## Dependencies


This repository requires the following modules:

* [Python 3](https://www.python.org/downloads/)
* Python libraries:
   * pandas>=0.25.1
   * numpy>=1.17.2
   * scikit-learn>=0.21.3
   * Theano>=1.0.4
   * tqdm>=4.36.1
   * pickle>=4.0


## USAGE

* cox_nnet_v2.py: This is a package for running the neural network extension of Cox regression

* HiddenNodes.py: This is for combing the iput to construct a new Cox-nnet model in the second stage

* test_2stage_opt.py: Two-stage cox-nnet model fitting

* test_ssBRCA_opt.py: Single-stage cox-nnet model fitting

* original_importance.ipynb: Deriving significant features among all CP, TMI, and TCI feature sets

### Execution
```bash
python3 HiddenNodes.py
```
## Maintainers

Current maintainers:
 * Shu Zhou - https://github.com/Sukumaru

