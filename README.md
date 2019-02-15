# dimensionality_reduction_PyData_2016
This talk provides a step-by-step overview and demonstration of several dimensionality (feature) reduction techniques.

* https://www.youtube.com/watch?v=ioXKxulmwVQ link of the video on dimensionality reduction
* https://scikit-learn.org/stable/modules/feature_selection.html feature selection in sklearn

## Why perform dimensionality reduction?
* with a fixed number of training samples, the predictive power reduces as the dimensionality increases.
* trade-off between predictive power and model interpretability.
* Law of Parsimony: simpler explanations are generally better than complex ones.
* prevent overfitting.
* lower execution time.

## Dimensionality reduction techniques:
1. Percent missing values
2. Amount of variation
3. Pairwise correlation
4. Multicollinearity
5. Principal Component Analysis (PCA)
6. Cluster analysis
7. Correlation with the target
8. Forward selection
9. Backward elimination (RFE)
10. Stepwise selection
11. LASSO
12. Tree-based selection

### 1. Percent missing values
A very simple approach, drop variables that have a high % of missing values. Missing values can have meaning, therefore create binary indicators to denote missing (or non-missing) values.
General rules of thumb:
* if more than 50% missing value, do not impute, provide binary indicators.
* if less than 50% missing value, perform imputation.
Easy to do in pandas: 
```python
import pandas as pd
desc = X.describe().T

#  percent missing
desc['missing %'] = 1 - (desc['count'] / len(X))
```


