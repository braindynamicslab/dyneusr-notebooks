dyneusr-notebooks
=================

[DyNeuSR-Notebooks](https://bitbucket.org/braindynamicslab/dyneusr-notebooks/) is a public repository of Jupyter notebook tutorials based on [DyNeuSR](https://bitbucket.org/braindynamicslab/dyneusr/).


Important links
===============

- Official DyNeuSR source code repo: https://bitbucket.org/braindynamicslab/dyneusr/
- Official DyNeuSR-Notebooks repo: https://bitbucket.org/braindynamicslab/dyneusr-notebooks/


Dependencies
============

These notebooks are based on [DyNeuSR](https://bitbucket.org/braindynamicslab/dyneusr/), and the required dependencies include: 

* [DyNeuSR](https://bitbucket.org/braindynamicslab/dyneusr/) == 0.1.0
* [Kepler-Mapper](https://github.com/MLWave/kepler-mapper) >= 1.1.5

For a full list of requirements, see: `requirements.txt`


Install
=======

To clone and install [DyNeuSR](https://bitbucket.org/braindynamicslab/dyneusr/), run the following commands in a command prompt:

	git clone https://bitbucket.org/braindynamicslab/dyneusr/ ~
	cd ~/dyneusr
	git checkout -b v0.1.0 tags/0.1.0
	pip install -e . -r requirements.txt

Note, you can replace the `~` with any path to install the `dyneusr` source code.

For more detailed installation instructions, see: https://bitbucket.org/braindynamicslab/dyneusr/README.md


Usage
=====

To view a Jupyter notebook example, you can run the following command in a command prompt:

	jupyter notebook [notebook.ipynb]


For example, to view the `01_trefoil_knot` notebook, run the following commands in a command prompt:

	cd 01_trefoil_knot
	jupyter notebook 01_trefoil_knot.ipynb


