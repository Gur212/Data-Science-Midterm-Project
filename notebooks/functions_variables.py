#Create function to open JSON files and place them into normalized dataframe
def load_json_file(file_path):
    
    import os
    import json
    import pandas as pd
    
    #open file
    with open(file_path, "r") as file:
        json_data = json.load(file)

    #Isolate "results" from json only
    json_results = json_data["data"]["results"]

    # Create dataframe
    df = pd.DataFrame(json_results)

    #Flatten nested dictionaries in data
    df = pd.json_normalize(df.to_dict(orient="records"))

    return df


#Create function to visualize all data points vs target variable
def vis_subplots(x_df, y_df,desired_columns=5, figure_size=[30,30]):
    #Visualize data to identify outliers
    from matplotlib import pyplot as plt
    import seaborn as sns
    import numpy as np
    import math

    #Define target variable name
    target_name = y_df.name.split(".")[-1]

    #Define subplot details
    total_plots = len(x_df.columns)
    total_rows = math.ceil(total_plots/desired_columns)
    rows, cols = total_rows, desired_columns
    fig, axes = plt.subplots(rows, cols, figsize=figure_size)
    axes = axes.flatten()

    #Create scatterplots for all columns in the training set and the target variable
    for i, column in enumerate(x_df.columns):
        #Try to create scatter plot if possible        
        try:
            sns.scatterplot(x=x_df[column], y=y_df, ax=axes[i])
            col_name = column.split(".")[-1]
            axes[i].set_xlabel(col_name)
            axes[i].set_ylabel(target_name)
            axes[i].set_title(f"{col_name} vs {target_name}")
        
        except: 
            print(f"Skipping column due to plotting error: {col_name}")

    # Hide any unused subplots (if features < 32)
    for j in range(total_plots, rows * cols):
        fig.delaxes(axes[j])

    #Show plots and avoid overlapping 
    plt.tight_layout()  
    plt.show()

def encode_tags(df, threshold):

    """Use this function to manually encode tags from each sale
    and filter out tags that appear less than threshold
       
    Args:
        pandas.DataFrame with a list of tags in 'tags'
        threshold for number of times the tag should appear

    Returns:
        pandas.DataFrame: modified with encoded tags from 'tags' column and for 'type'
    """
    from collections import Counter
    
    #Iterate over 'tags', and each sublist in it to create a list of all tags
    all_tags = [tag for sublist in df['tags'].dropna() for tag in sublist]

    #Count number of times each tag appears in list 
    tag_counts = Counter(all_tags) 

    #Create a dictionary of tags that appear more than the threshold input
    valid_tags = {tag for tag, count in tag_counts.items() if count >= threshold}

    #If df['tags'] is a list, iterate over it and see if it's in valid_tags. If it is, it is put into a new list of 'filtered_tags'
    df['filtered_tags'] = df['tags'].apply(lambda tag_list: [tag for tag in tag_list if tag in valid_tags] if isinstance(tag_list, list) else [])

    #Explodes the list of filtered tags, turns them into strings, encodes them, and then combines the exploded tags back
    ohe_df = df['filtered_tags'].explode().str.get_dummies().groupby(level=0).sum()

    #Drop columns already accounted for. Specific to this data
    ohe_df = ohe_df.drop(columns=['garage_1_or_more', 'garage_2_or_more', 'single_story', 'two_or_more_stories'])
    
    #Join OHE dataframe and drop tags column
    df = df.drop(columns=['tags']).join(ohe_df)

    #Drop 'filtered_tags'
    df = df.drop(columns=['filtered_tags'])

    #OHE for description.type
    type_ohe_df = df['description.type'].str.get_dummies()

    #Join type_OHE dataframe and drop description.type column
    df = df.drop(columns=['description.type']).join(type_ohe_df)
    
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

    Returns: 
        None
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
def find_best_regression_model (X_train, X_test, y_train, y_test):
    """
    Use this function to try many different linear regression models to find the r2 scores and rank them. This function attempts to use 7 linear regression models:
    Linear Regression, Elastic Net, Decision Tree, Random Forest, XG Boost, LightGBM & Neural Networks

    Args:
        X_train: Independant variable training dataframe
        X_test: Independant test dataframe
        y_train: Target variable training dataframe
        y_test: Target variable test dataframe

    Returns: 
        None
    """
    #import needed modules
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import ElasticNet

    #Suppress convergance warnings
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)


    #List models to discover
    models = {
        "Linear Regression": LinearRegression(),
        "Elastic Net": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(),
        "LightGBM": LGBMRegressor(verbose=0),
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(64, 32), 
            n_iter_no_change=10, 
            max_iter=2000,
            learning_rate_init=0.001
            )
    }

    #Empty results dictionary
    results = {}

    #Convert both y_train and y_test to 1D arrays
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()
    
    # Train and evaluate models
    for name, model in models.items():
        print(f"Processing {name}...")

        model.fit(X_train, y_train)

        #Make predictions    
        iy_pred = model.predict(X_test)

        #Calculate error scores
        r2 = round(r2_score(y_test, iy_pred),4)
        mse = round(mean_squared_error(y_test, iy_pred),2)
        mae = round(mean_absolute_error(y_test, iy_pred),2)
        
        #Add results to dictionary
        results[name] = {
            "R² Score": r2,
            "MSE": mse, 
            "MAE": mae
        }

    # Convert results to DataFrame for better visualization
    results_df = pd.DataFrame(results).T
    results_df_sorted = results_df.sort_values(by='R² Score', ascending=False)
    results_df_sorted['Rank'] = results_df_sorted['R² Score'].rank(method='dense', ascending=False).astype(int)

    print(f"Processing COMPLETE!")

    return results_df_sorted

#Function to cross-validate data without leaking when Target Encoding
def custom_cross_validation(training_data, splits =5):
    '''creates n_splits sets of training and validation folds

    Args:
      training_data: the dataframe of features and target to be divided into folds
      n_splits: the number of sets of folds to be created

    Returns:
      A tuple of lists, where the first index is a list of the training folds, 
      and the second the corresponding validation fold

    Example:
        >>> output = custom_cross_validation(train_df, n_splits = 10)
        >>> output[0][0] # The first training fold
        >>> output[1][0] # The first validation fold
        >>> output[0][1] # The second training fold
        >>> 
    '''
    
    from sklearn.model_selection import KFold
    
    kfold = KFold(n_splits=splits, shuffle=True)
    train_folds = []
    val_folds = []
    
    for training_index, val_index in kfold.split(training_data):
        #Creating dataframes for training and validation folds
        train_fold = training_data.iloc[training_index]
        val_fold = training_data.iloc[val_index]
    
        #Setting up dictionary where the key is each city/state in the training fold, and the value is the mean
        default_mean = train_fold['description.sold_price'].mean()       #In case there is a city/state present in validation fold but not in training
        cities = train_fold['location.address.city'].unique().tolist()
        city_dict = {key:None for key in cities}
        states = train_fold['location.address.state'].unique().tolist()
        state_dict = {key:None for key in states}
        
        #Filling the values of the dictionary with means 
        for key in city_dict.keys():
            city_dict[key] = train_fold.loc[train_fold['location.address.city'] == key, 'description.sold_price'].mean()
    
        for key in state_dict.keys():
            state_dict[key] = train_fold.loc[train_fold['location.address.state'] == key, 'description.sold_price'].mean()
    
        #Iterate over the training fold and replace the city/state for the mean 
        for index, rows in train_fold.iterrows():
            city = train_fold['location.address.city'].at[index]
            state = train_fold['location.address.state'].at[index]
            train_fold.at[index, 'location.address.city'] = city_dict[city]
            train_fold.at[index, 'location.address.state'] = state_dict[state]

        #Iterate over the validation fold and replace the city/state for the mean values calculated from the training fold
        #If the city/state is not in the training fold, it's replaced by the default_mean
        for index, rows in val_fold.iterrows():
            city = val_fold['location.address.city'].at[index]
            try:
                val_fold.at[index, 'location.address.city'] = city_dict[city]
            except:
                val_fold.at[index, 'location.address.city'] = default_mean
                
            state = val_fold['location.address.state'].at[index]
            try:   
                val_fold.at[index, 'location.address.state'] = state_dict[state]
            except:
                val_fold.at[index, 'location.address.state'] = default_mean
        '''
        train_fold['location.address.city'] = train_fold['location.address.city'].astype(float)
        train_fold['location.address.state'] = train_fold['location.address.state'].astype(float)
        val_fold['location.address.city'] = val_fold['location.address.city'].astype(float)
        val_fold['location.address.state'] = val_fold['location.address.state'].astype(float)
        '''
        
        train_fold = train_fold.astype(float)
        train_fold = train_fold.astype(float)
        val_fold = val_fold.astype(float)
        val_fold = val_fold.astype(float)
        
        train_folds.append(train_fold)
        val_folds.append(val_fold)
        
    return train_folds, val_folds


#Function that takes a parameter dictionary and output from custom_cross_validation, and returns the best parameter combo
def hyperparameter_search(output, param_grid):

    from itertools import product
    from xgboost import XGBRegressor
    from sklearn.metrics import root_mean_squared_error
    from statistics import mean

    #Setting list of dataframes for training_folds and validation_folds
    training_folds=output[0]
    validation_folds=output[1]

    #Takes the values in param_grid and returns the itertools.product (list of every possible combination of parameters)
    hyperparams = list((dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())))
    hyperparam_scores = []
    xg = XGBRegressor()

    #Iterates over list of combos of hyperparameters
    for combo in hyperparams:
        fold_scores = []
        #iterates over the number of folds
        for fold in range(len(training_folds)):
            #Creating test/train splits from each fold
            X_train = training_folds[fold].drop(columns='description.sold_price')
            y_train = training_folds[fold]['description.sold_price']
            X_test = validation_folds[fold].drop(columns='description.sold_price')
            y_test = validation_folds[fold]['description.sold_price']
            
            #Fitting model
            xg.fit(X_train, y_train)
            y_pred = xg.predict(X_test)

            #Finding the RMSE the param combo
            score_fold = root_mean_squared_error(y_test, y_pred)
            fold_scores.append(score_fold)

        #taking the mean of the RMSE for all the folds for each param combo
        score = mean(fold_scores)
        hyperparam_scores.append(score)

    #finding the param combo with the lowest RMSE
    index_min = min(range(len(hyperparam_scores)), key=hyperparam_scores.__getitem__)
    best_params = hyperparams[index_min]

    print(f'Parameters with lowest RMSE: {best_params}')