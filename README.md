# DyNeuSR Notebooks
-------------------

DyNeuSR Notebooks is a collection of Jupyter notebook tutorials based on [DyNeuSR](https://bitbucket.org/braindynamicslab/dyneusr/).

From the [DyNeuSR/REAME.md](https://bitbucket.org/braindynamicslab/dyneusr/src/master/README.md):

> DyNeuSR is a Python visualization library for topological representations of neuroimaging data. It was designed specifically for working with shape graphs produced by the Mapper algorithm from topological data analysis (TDA) as described in the paper ["Towards a new approach to reveal dynamical organization of the brain using topological data analysis"](https://www.nature.com/articles/s41467-018-03664-4) (Saggar et al., 2018). See this [blog post](https://bdl.stanford.edu/blog/tda-cme-paper/) for more about the initial work that inspired the development of DyNeuSR.  
>
> DyNeuSR connects the Mapper algorithm (e.g., [KeplerMapper](https://kepler-mapper.scikit-tda.org)), with network analysis tools (e.g., [NetworkX](https://networkx.github.io/)) and other neuroimaging data visualization libraries (e.g., [Nilearn](https://nilearn.github.io/)). It provides a high-level interface for interacting with shape graph representations of neuroimaging data and relating such representations back to neurophysiology.


## Usage
--------

To view a Jupyter notebook example, you can run the following command in a command prompt:
```bash
jupyter notebook [notebook.ipynb]
```

For example, to view the `01_trefoil_knot` notebook, just run:
```bash
jupyter notebook 01_trefoil_knot/01_trefoil_knot.ipynb
```



## Install
----------

### Dependencies

- Python 3.6+

The latest versions of the following packages are required:

-  [dyneusr](https://bitbucket.org/braindynamicslab/dyneusr/) == 0.2.3
-  [kmapper](kepler-mapper.scikit-tda.org) == 1.2.0

For the full list, see [`requirements.txt`](https://bitbucket.org/braindynamicslab/dyneusr-notebooks/src/master/requirements.txt)


### Installation

To clone and install [DyNeuSR](https://bitbucket.org/braindynamicslab/dyneusr/):
```bash
git clone https://bitbucket.org/braindynamicslab/dyneusr.git
cd dyneusr
git checkout -b v0.2.3 tags/0.2.3
pip install -e .
```

For more detailed installation instructions, see [DyNeuSR/README.md](https://bitbucket.org/braindynamicslab/dyneusr/src/master/README.md)




## Support
----------

Please [submit](https://bitbucket.org/braindynamicslab/dyneusr-notebooks/issues/new) any bugs or questions to the Bitbucket issue tracker.



## Cite
-------

(*coming soon*)

