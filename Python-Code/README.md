# Code Description
A brief overview of the steps taken by the Code

## Data processing
1. change SEX from string to Numeric
- male=0 female=1
2. delete from from endpoint column
- each endpoint has 3 columns
 - endpoint, endpoint_AGE, endpoint_NEVT
- the first plain column is dropped, because the information is also available in NEVT
3. in addition, BL_AGE (the age at which the blood sample was taken) is deleted, because this should not affect diseases
4. delete the non-related age
- endpoint_AGE contains _AGE at the endpoint entry or investigation ends
- End of investigation is not decisive and falsifies important data from the age at the entry of the endpoint.
-Therefore all values that are not connected to an endpoint are removed
5. pill subscription data is merged with endpoint data via FINGENID column
6. all columns are dropped that contain only zeros


## Visualisation

## Machine Learning - Decision Tree


