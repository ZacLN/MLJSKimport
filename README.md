# Code gen for MLJ/ScikitLearn interaction
Pulls model specification info from https://raw.githubusercontent.com/alan-turing-institute/MLJ.jl/master/material/sklearn_reggressors.csv constructing corresponding datatypes/methods in Julia:
- imports ScikitLearn PyObject
- creates equivalent Julia struct
- creates default constructor
- creates `fit` method
- creates `predict` method

### Todo
- Improve parsing of kw default values
- Fill in `report`s