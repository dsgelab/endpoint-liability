# Code Description
A brief overview of the steps taken by the Code

## Data processing
1. Change Sex from string to Numeric
- male=0 female=1
2. Delete plain endpoint columns
- Each endpoint has 3 columns
    - Endpoint, endpoint_AGE, endpoint_NEVT
- The first plain column is dropped, because the information is also available in NEVT
3. In addition, BL_AGE (the age at which the blood sample was taken) is deleted, because this should not affect diseases
4. Pill subscription data is merged with endpoint data via FINGENID column
5. Parse everything to numeric
6. Iteration through each column
    1. If a Column contains just one variable save Column name in drop list
    2. Delete the non-related age
        - Endpoint_AGE contains _AGE at the endpoint entry or investigation ends
        - End of investigation is not decisive and falsifies important data from the age at the entry of the endpoint.
        - Therefore all values that are not connected to an endpoint are removed
7. All columns are dropped that contain more than 99.5% zeros
8. Deleting drop list


## Visualisation

## Machine Learning - Decision Tree
1. Checks if other endpoints are highly correlated to endpoint of interest (spearman correlation)
- writes column name in droplist if there is a correlation of 99.5
2. Deletes highly correlated endpoint and medical related endpoints
3. Sets independent and dependent Variable
4. train xgboost Tree

## Packages Used
<p> import numpy as np <br/>
import pandas as pd <br/>
import seaborn as sns <br/>
import os <br/>
import sys <br/>
import math <br/>
import re <br/>
import xgboost as xgb <br/>
import sklearn </p>
