# Single-cell based prognosis modeling identifies new breast cancer survival subtypes by cell-cell interactions

## Description

This is the github repository for the manuscript **Single-cell based prognosis modeling identifies new breast cancer survival subtypes by cell-cell interactions** by Shashank Yadav, Shu Zhou, Bing He and Lana Garmire et al.. It contains code and data for generating Figure 1-4,6 and 7 in the manuscript. 

## Getting Started

### Dependencies
* Linux Working Environment
* [Python 3](https://www.python.org/downloads/)
* [R](https://www.R-project.org)
* [Anaconda3](https://www.anaconda.com/)
* [Jupyter](https://jupyter.org)
* Python libraries:
  * [Numpy](https://numpy.org/)
  * [pandas](https://pandas.pydata.org/docs/index.html)
  * [Scipy](https://scipy.org/)
  * [Scikit-learn](http://scikit-learn.org/)
  * [coxnnet](http://garmiregroup.org/cox-nnet/docs/)
  * [Theano](https://github.com/Theano/Theano)
  * [tqdm](https://github.com/tqdm/tqdm)
  * [pickle](https://docs.python.org/3/library/pickle.html)
  * [seaborn](https://seaborn.pydata.org/)
  * [holoviews](https://holoviews.org/)
  * [plotly](https://plotly.com/)
  * [lifelines](https://lifelines.readthedocs.io/en/latest/)
  * [UnionCom](https://github.com/caokai1073/UnionCom)
* R libraries:
  * [NMF](https://cran.r-project.org/web/packages/NMF/index.html)
  * [circlize](https://github.com/jokergoo/circlize)
  * [BBmisc](https://cran.rstudio.com/web/packages/BBmisc/index.html)


### Installing

Installing the R kernel on the jupyter
```R
install.packages('IRkernel')
IRkernel::installspec()  # to register the kernel in the current R installation
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install python packages.
```bash
pip install Numpy
```

## Repository directories & files

The directories are as follows:
+ [`1_HMResult`](1_HMResult) contains the heatmaps in Figure.1 of the manuscript.
+ [`2_PrognosisPred`](2_PrognosisPred) contains the modeling of prognosis prediction using single-cell phenotype features, constructing cox-nnet models
+ [`4_SankeyResult`](4_SankeyResult) contains SankeyPlots descrbing how the NMF-defined classes intersect with the clinicopathological classification
+ [`7_AtypicalMatching`](7_AtypicalMatching) contains class label transfer and comparison with TCGA-BRCA and METABRIC datasets.
+ [`Figures`](Figures) contains Figure 1-7 showing in the manuscript.
+ [`Pydata`](Pydata) contains python data used for [`4_sankey.ipynb`](4_sankey.ipynb) and [`6_Violin.ipynb`](6_Violin.ipynb)
+ [`rds`](rds) contains `.rds` data used for [`1_HMOverview.ipynb`](1_HMOverview.ipynb), [`3_NMFPlot.ipynb`](3_NMFPlot.ipynb) and [`4_HMCircos.ipynb`](4_HMCircos.ipynb)

The other files are as follows.
+ [`1_HMOverview.ipynb`](1_HMOverview.ipynb) contains the ploting process of the general data heatmaps.
+ [`2_ProgPlot.ipynb`](2_ProgPlot.ipynb) contains the values in combining different sets of information from CP, TMI, and TCI.
+ [`3_NMFPlot.ipynb`](3_NMFPlot.ipynb) contains the result of NMF clustering and the consensusmap.
+ [`3_NMFScore.ipynb`](3_NMFScore.ipynb) contains the visualization procedure of the silhouette and cophenetic score
+ [`3_kaplanmeier_DFS.ipynb`](3_kaplanmeier_DFS.ipynb) contains the kaplanmeier plot for NMF clustering and Grade, ER, PR, HER2 types.
+ [`4_HMCircos.ipynb`](4_HMCircos.ipynb) contains the Heatmaps of Grade, ER, PR, HER2 and cell-cell interaction features for the NMF clusters and Circos plots demonstrate the correlation between features associated with each subpopulation.
+ [`4_sankey.ipynb`](4_sankey.ipynb)contains procedures of constructing sankey plots
+ [`6_Violin.ipynb`](6_Violin.ipynb)contains scoring and profiling for the seven clusters based on various Cell phenotypes and Cell-Cell interaction features.

### Local execution
+ for `.ipynb` files: Using jupyter lab to execute the codes
+ for `.py` files:
```bash
python3 -m *filename*.py
```

## Authors

### Contributors names and contact info

+ [@Lana Garmire](https://github.com/lanagarmire)
+ [@Shashank Yadav](https://github.com/xinformatics)
+ [@Bing He](https://github.com/hebinghb)
+ [@Shu Zhou](https://github.com/Sukumaru)

### Current Maintainer
* Shu Zhou - https://github.com/Sukumaru

## License

This project is licensed under the `GNU General Public License v3.0` License - see the LICENSE.md file for details
