## Description

This part refers to the:

*Class label transfer and comparison with TCGA-BRCA and METABRIC datasets* 

part of the project



## Dependencies


This repository requires the following modules:

* [Python 3](https://www.python.org/downloads/)
* Python libraries:
   * pandas
   * numpy
   * scikit-learn
   * lifelines

* [R 4.2.0](https://www.r-project.org/)
   * limma
   * Biobase
   * ggrepel
   * magrittr
   * tidyverse
   * ggplot2


## Repository directories & files
The files are as follows:
* [`HD-KM-PLots.ipynb`](HD-KM-PLots.ipynb): Kaplan-Meier Plot for the TNBC and LumA subtypes for TCGA-BRCA and METAABRIC dataset.

* [`label_transfer.ipynb`](label_transfer.ipynb): Transferring labels from Basel data to TCGA-BRCA and METAABRIC dataset

* [`limma_METABRIC.ipynb`](limma_METABRIC.ipynb): Differential analysis of METABRIC dataset on TNBC and LumA subtypes

* [`limma_TCGA.ipynb`](limma_TCGA.ipynb): Differential analysis of TCGA dataset on TNBC and LumA subtypes

* [`limma_ours.ipynb`](limma_ours.ipynb): Differential analysis of the dataset investigated in the project (Basel) on TNBC and LumA subtypes

The directories are as follows:
* [`KM-PLots`](KM-PLots): Result of Kaplan-Meier Plots for the TNBC and LumA subtypes for TCGA-BRCA and METAABRIC dataset.
* [`cli`](cli): Clinical data of TCGA, METABRIC and BRCA dataset
* [`exp`](exp): TNBC and Lum-A subtypes data of TCGA-BRCA and METABRIC dataset
* [`limmaPlots`](limmaPlots) Differential analysis result for TCGA-BRCA and METAABRIC dataset.

### Execution

Using the anaconda prompt to open the `.ipynb` files

## Maintainers

Current maintainers:
 * Shu Zhou - https://github.com/Sukumaru
