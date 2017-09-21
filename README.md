# Generalized Approximated Leave-One-Out Cross-Validation

This package implements the generalized approximated leave-one-out cross-validation (GALO) in Python.

Regarding the structure
* This package should mainly focus on ALO part. Since ALO itself has nothing to do with those optimization directly, the input should assume the input is provided;
* On the other hand, in order to calculate ALO, we should also konw information about loss and regularizers. There should be a parent class that implement general ALO (or we can summarize general ALO into 2 or 3 categories and define a parent class for each type). _Think_ in what form should we pass the infomation about loss and regularizers (mainly their derivatives) to these class. Then for child class, we can provided more detailed choices, in string, for specific statistical problems, such as GLM, matrix completion, SLOPE, etc.
* Maybe, the parent class should be implemented simply in C functions, to light-weight it, and they wrap it in child class.
* Think about what should be appropriate test cases;
* Documentation.
* Ask Arian the permission to releash the code.
