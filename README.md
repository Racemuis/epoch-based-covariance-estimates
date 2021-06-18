# Epoch-based Covariance Matrices
This is the repository where I store the code that I have used for my Bachelor thesis. This repository includes files with which I have generated the figures for my thesis and my presentation, as well as the source code of my methods and results sections. 

## Installation
The source code for this project is written in python. To let anyone obtain the same results as presented in the thesis, I have exported my conda environment with the required dependencies and versions. To install this enviroment using conda, one can download the `environment.yml` file and create the environment in the anaconda prompt using `conda env create -f environment.yml`.

## Usage
### Figures & Analyses
All figures and analyses are standalone files. To use them, one can change the `./path/to/file` placeholders with the desired trajectory.

### Methods
To run the analyses that correspond to the _Methods_ Chapter of the thesis, one can run the file `methods/main_moabb_pipeline.py` in a python command line. The file is depedent on the local configurations that are set up in `methods/configurations/local_config.yaml`, hence, ensure that this file is filled in. Next, `main_moabb_pipeline.py` takes 3 arguments: 
1. The dataset, possible datasets are:
    - spot_single
    - epfl
    - braininvaders
    - bnci_1
    - bnci_2
    - bnci_als
   If the datasets are not available, they are downloaded automatically.
2. The subjects
3. The sessions

Hence, the file can be run by typing `python main_moab_pipeline.py <dataset> <subject:session>`.<br>
E.g. `python main_moab_pipeline.py spot_single 0:0` for the first session of the first subject of the SPOT dataset. <br>

_Note that it is important to activate the right conda environment prior to running the command above. This can be done using_ `conda activate thesis`.<br>

## References & Licenses
My methods are based on the paper listed at [1]. The source files that accompany the paper are released under the MIT license. They can be found in the [following repository](https://github.com/jsosulski/time-decoupled-lda). Moreover, this project relies heavily on the [pyRiemann library](https://zenodo.org/record/18982#.YLdhSKgzZPY) [2] and the [Mother of All BCI Benchmarks (MOABB)](https://github.com/NeuroTechX/moabb) [3]. As the firstmentioned source files did not come in a pip-able package, I had to copy some of the files to my own repository. The license has been reproduced in the corresponding source files. 


[1] J.  Sosulski,  J.  Kemmer,  and  M.  Tangermann,  “Improving  Covariance  Matrices  Derivedfrom Tiny Training Datasets for the Classification of Event-Related Potentials with LinearDiscriminant Analysis,”_Neuroinformatics_, pp. 1–16, 2020.<br>
[2] M. Congedo, A. Barachant, and R. Bhatia, “Riemannian geometry for EEG-based brain-computer  interfaces;  a  primer  and  a  review,” _Brain-Computer  Interfaces_,  vol.  4,  no.  3,pp. 155–174, 2017.<br>
[3] V. Jayaram and A. Barachant, “MOABB: trustworthy algorithm benchmarking for BCIs,” _Journal of Neural Engineering_, vol. 15, no. 6, pp. 1741–2552, 2018.<br>
