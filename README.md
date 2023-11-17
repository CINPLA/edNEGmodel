[![Build Status](https://app.travis-ci.com/CINPLA/edNEGmodel.svg?branch=master)](https://app.travis-ci.com/CINPLA/edNEGmodel)

# edNEGmodel 

edNEGmodel is an implementation of the KNP continuity equations for a
one-dimensional system containing six compartments: two representing a neuron, two representing extracellular space, and two representing glia.
The model is presented in Sætra et al., *PLoS Computational Biology*, 17(7), e1008143 (2021): 
[An electrodiffusive neuron-extracellular-glia model for exploring
the genesis of slow potentials in the brain](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008143).

## Installation 

Clone or download the repo, navigate to the top directory of the repo and enter the following
command in the terminal: 
```bash
python3 setup.py install
```

The code was developed for Python 3.6.

## Run simulations

The simulations folder includes example code showing how to run simulations. 
To reproduce the results presented in Sætra et al. 2021, see 
[https://github.com/CINPLA/edNEGmodel_analysis](https://github.com/CINPLA/edNEGmodel_analysis).

## Errata

* Table 1 in Sætra et al. 2021 says that the extracellular cross-section area A<sub>e</sub> is 3.08e-11 m<sup>2</sup>. 
However, the value used in our simulations is 6.16e-11 m<sup>2</sup>. 
Thanks to Eirill Hauge and Letizia Signorelli for making us aware of this error.
