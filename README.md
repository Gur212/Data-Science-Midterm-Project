# Data Science Midterm Project
by Jeevan and James

## Project/Goals
To train and tune a machine learning model that is able to predict the sale price of a house based on certain factors
## Process
- #### Data Cleaning and Tranformation
    - parsed data and entered it into a dataframe to be manipulated 
    - removed features that weren't of use for our goals
    - removed entries with missing data that could not be interpolated
    - calulated values for missing data that could be interpolated 
    - converted features like date into a more appropriate format
    - scaled data to a standardized distribution
    - employed one hot encoding for categorical data
    - used targeted encoding for locational data

- #### Model Selection and Feature Engineering
    - trained and fitted multiple machine learning models to see which performed best using default hyperparameters
    - performed feature selection to reduce dimensionality of chosen model
    - 
- #### Hyperparameter Tuning
    - used GridSearchCV to find optimal hyperparameters 
    - created user defined functions to perform cross validation and hyperparameter searching while avoiding data leakage
    - refit model with tuned hyperparameters

## Results
The model we ended up selecting was XGBoost. With our minimum viable model, we were able to achieve an R <sup>2</sup> score of 0.994 on our test data. We got a mean average error (MAE) of $8567.57 and a root mean squared error (RMSE) of $14489.44. After performing a custom cross validation to minimise data leakage, we got very similar results with an R <sup>2</sup> score of 0.994, an MAE of $8104.39 and an RMSE of $14282.57.

## Challenges 
Even though the data appeared to be relatively clean, there was a lot of processing that had to be performed on it to make it usable.

## Future Goals
If we had more time, we'd like to explore how well we could tune each model to perform. It's also possible that there was leakage from the data wrangling, so we'd like to explore splitting the data earlier on. 
