# Single-cell based prognosis modeling identifies new breast cancer survival subtypes by cell-cell interactions

## Description

This is the github repository for the manuscript **Single-cell based prognosis modeling identifies new breast cancer survival subtypes by cell-cell interactions** by Shashank Yadav, Bing He, and Shu Zhou et al.. It contains code and data for generating Figure 1-4,6 and 7 in the manuscript. 

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

### Repository directories & files

The directories are as follows:
+ [`1_HMResult`](1_HMResult) contains the heatmaps in Figure.1 of the manuscript.
+ [`2_PrognosisPred`](2_PrognosisPred) contains the modeling of prognosis prediction using single-cell phenotype features, constructing cox-nnet models
+ [`4_SankeyResult`](4_SankeyResult) cpntains SankeyPlots descrbing how the NMF-defined classes intersect with the clinicopathological classification
+ [`7_AtypicalMatching`](7_AtypicalMatching) contains class label transfer and comparison with TCGA-BRCA and METABRIC datasets.
+ [`Figures`](Figures) contains Figure 1-7 showing in the manuscript.
+ [`Pydata`](Pydata) contains python data used for [`6_violin`](6_violin.ipynb)

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the `GNU General Public License v3.0` License - see the LICENSE.md file for details
