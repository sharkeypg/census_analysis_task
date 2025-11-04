# Census income analysis

This repository contains two main components:

```
src/
	config.py		## defined parameters for cleaning and modelling
	data_validation.py 	## some functions for working with the data schema
	model.py		## functions for data cleaning, feature engineering, modelling and evaluation

notebooks/
	EDA.ipynb		## and end-to-end exploratory analysis notebook
	model.ipynb		## a notebook to run the different models explored
```

To run either notebook, create a new environment, e.g.

```
python -m venv census_task  
```

Activate that environment:

```
source census_task/bin/activate
```

Install packages in requirements file:

```
pip install -r requirements.txt
```

Set your kernel to be the Python environment when running the notebook. This work was developed using Python version 3.12.2.

Note that the functions developed in this repository were written with testing in mind, but due to time constraints, unit testing has not been implemented.

The datasets have not been provided with the task submission due to their size. Please add the following to the `data/` directory: `census_income_learn.csv`, `census_income_test.csv` and `census_income_metadata.txt`.
