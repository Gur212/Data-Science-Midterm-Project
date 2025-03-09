def encode_tags(df):

    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low 
    counts of tags to keep cardinality to a minimum.
       
    Args:
        pandas.DataFrame

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
    tags = df["tags"].tolist()
    # create a unique list of tags and then create a new column for each tag
        
    return df

#Create function to get all error scores at once
def get_error_scores (y_train, y_train_pred, y_test, y_test_pred, error_type='All', num_results=10):
    """
    Use this function to get error scores for a desired model

    Args:
        y_train: Dataframe of training data target variables
        y_train_pred: Dataframe or array of predicted variables on training data
        y_test: Dataframe of test data target variable
        y_test_pred: Dataframe or array of predicted variabel on test data
        error_type: (Optional) Score desired for return. Default = 'All' Options include r2, mae & rmse
        num_results: (Optional) The number of results desired in return. Default = 10

    Returns: 
        None
    """
    
    # Check performance on train and test set
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import numpy as np

    if (error_type == 'All' or LOWER(error_type) == 'r2'):
        #Using R2
        r2_train = round(r2_score(y_train, y_train_pred),4)
        r2_test = round(r2_score(y_test, y_test_pred),4)

        print(f'R SQUARED\n\tTrain R²:\t{r2_train}\n\tTest R²:\t{r2_test}')

    if (error_type == 'All' or LOWER(error_type) == 'mae'):
        #Using Mean Average Error
        MAE_train = round(mean_absolute_error(y_train, y_train_pred),2)
        MAE_test = round(mean_absolute_error(y_test, y_test_pred),2)

        print(f'MEAN AVERAGE ERROR\n\tTrain MAE:\t{MAE_train}\n\tTest MAE:\t{MAE_test}')

    if (error_type == 'All' or LOWER(error_type) == 'rmse'):
        #Using Root Mean Squared Error
        RMSE_train = round(np.sqrt(mean_squared_error(y_train, y_train_pred)),2)
        RMSE_test = round(np.sqrt(mean_squared_error(y_test, y_test_pred)),2)

        print(f'ROOT MEAN SQUARED ERROR\n\tTrain RMSE:\t{RMSE_train}\n\tTest RMSE:\t{RMSE_test}\n')

    if (error_type == 'All'):
        display_results_sample(y_test, y_test_pred, num_results)

#Function to create a demonstrate of prediction
def display_results_sample (y_test, y_test_prediction, num_results=10):
    """
    Use this function to get a random sample of predictions and compare to actual results

    Args: 
        y_test: Dataframe of test data target variable
        y_test_pred: Dataframe or array of predicted variabel on test data
        num_results: (Optional) The number of results desired in return. Default = 10

    """
    import random
    import numpy as np

    print(f"{num_results} Randomly selected results.")

    sum_percentage_error = 0

    #Choose 10 rows to display
    for i in range(num_results):
        j = random.randint(0, len(y_test)-1)

        if isinstance(y_test_prediction[j], (list, tuple, np.ndarray)):
            demo_prediction = round(y_test_prediction[j][0])
        else:
            demo_prediction = round(y_test_prediction[j])
        demo_actual = round(y_test.iloc[j].item())
        demo_difference = demo_prediction - demo_actual
        demo_difference_percentage = round((demo_actual / demo_prediction - 1)*100,2)

        sum_percentage_error += abs(demo_difference_percentage)

        print(f"Index: {j} \t- \tPrediction: ${demo_prediction:,} \tActual: ${demo_actual:,} \tDifference: {demo_difference:,}, {demo_difference_percentage}%")

    average_percentage_error = round(sum_percentage_error / num_results,2)
    print(f"\t\t\t\t\t\t\t\t\tAverage % error = {average_percentage_error}%")

#Function to find best linear regression model
def find_best_regression_model (iX_train, iX_test, iy_train, iy_test):
    #import needed modules
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import ElasticNet


    #List models to discover
    models = {
        "Linear Regression": LinearRegression(),
        "Elastic Net": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(),
        "LightGBM": LGBMRegressor(),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500)
    }

    #Empty results dictionary
    results = {}

    # Train and evaluate models
    for name, model in models.items():
        print(f"Processing {name}")
        model.fit(iX_train, iy_train)
        iy_pred = model.predict(iX_test)
        mse = mean_squared_error(iy_test, iy_pred)
        r2 = r2_score(iy_test, iy_pred)
        
        results[name] = {"MSE": mse, "R² Score": r2}

    # Convert results to DataFrame for better visualization
    results_df = pd.DataFrame(results).T
    results_df_sorted = results_df.sort_values(by='R² Score', ascending=False)

    print(f"Processing COMPLETE!")

    return results_df_sorted