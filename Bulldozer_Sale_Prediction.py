# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 13:04:09 2022
@author: Araz
Predicting the Sale Price of Bulldozers using Machine Learning
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

df =  pd.read_csv('data/TrainAndValid.csv',low_memory=False,parse_dates=['saledate'])
df_tmp = df.copy() 
# %%
print(df.info(),'\n\n')
print(df.isna().sum())

plt.scatter(df['saledate'][:1000],df['SalePrice'][:1000])
plt.xlabel('Sale Date')
plt.ylabel('Sale Price')
plt.show()

df_tmp.sort_values(by=["saledate"], inplace=True, ascending=True)
df_tmp["saleYear"] = df_tmp.saledate.dt.year
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saleDayOfWeek"] = df_tmp.saledate.dt.dayofweek
df_tmp["saleDayOfYear"] = df_tmp.saledate.dt.dayofyear

df_tmp.drop("saledate", axis=1, inplace=True)

# %%
'''
print(df_tmp.state.value_counts(),'\n\n')

#Finding String type Data

for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)
'''
# %%
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()

df_tmp.to_csv("data/train_tmp.csv",
              index=False)

df_tmp = pd.read_csv("data/train_tmp.csv",
                     low_memory=False)

# %%

'''
# Check which columns are numeric 
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)

print('\n\n')

# Check for which numeric columns have null values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)
'''

# Fill numeric rows with the median
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column which tells us if the data was missing or not
            df_tmp[label+"_is_missing"] = pd.isnull(content)
            # Fill missing numeric values with median
            df_tmp[label] = content.fillna(content.median())


# Turn categorical variables into numbers and fill missing
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to indicate whether sample had missing value
        df_tmp[label+"_is_missing"] = pd.isnull(content)
        # Turn categories into numbers and add +1
        df_tmp[label] = pd.Categorical(content).codes+1

#%%
#This cell takes a time to run
'''
#from datetime import datetime
#start = datetime.now()

# Instantiate model
model = RandomForestRegressor(n_jobs=-1,random_state=42) # random state so our results are reproducible

# Fit the model
model.fit(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])

#print(datetime.now() - start)

print(model.score(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"]))
'''
#%%

# Split data into training and validation

df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]

X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice

# Create evaluation function (the competition uses RMSLE)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_test, y_preds):
    """
    Caculates root mean squared log error between predictions and
    true labels.
    """
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluate model on a few different levels
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),
              "Valid MAE": mean_absolute_error(y_valid, val_preds),
              "Training RMSLE": rmsle(y_train, train_preds),
              "Valid RMSLE": rmsle(y_valid, val_preds),
              "Training R^2": r2_score(y_train, train_preds),
              "Valid R^2": r2_score(y_valid, val_preds)}
    return scores

#%%

# Change max_samples value to reduce model fitting time

model = RandomForestRegressor(n_jobs=-1, random_state=42, max_samples=10000)
model.fit(X_train, y_train)

print(show_scores(model))

#%%
'''
from datetime import datetime
start = datetime.now()

# Different RandomForestRegressor hyperparameters
rf_grid = {"n_estimators": np.arange(10, 100, 10),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2),
           "max_features": [0.5, 1, "sqrt", "auto"],
           "max_samples": [10000]}

# Instantiate RandomizedSearchCV model
rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,random_state=42),
                              param_distributions=rf_grid,
                              n_iter=10,
                              cv=5,
                              verbose=True)

# Fit the RandomizedSearchCV model
rs_model.fit(X_train, y_train)

print(datetime.now() - start)

print(rs_model.best_params_)
print(show_scores(rs_model))
'''
#%%

#Best hyperparamters found with n_iter=100

ideal_model = RandomForestRegressor(n_estimators=40,
                                    min_samples_leaf=1,
                                    min_samples_split=14,
                                    max_features=0.5,
                                    n_jobs=-1,
                                    max_samples=None,
                                    random_state=42) 

# Fit the ideal model
ideal_model.fit(X_train, y_train)

print(show_scores(ideal_model))

#%%

#Make predictions on test data

df_test = pd.read_csv("data/Test.csv",
                      low_memory=False,
                      parse_dates=["saledate"])


#Preprocessing the data (getting the test dataset in the same format as our training dataset)

def preprocess_data(df):
    """
    Performs transformations on df and returns transformed df.
    """
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeek"] = df.saledate.dt.dayofweek
    df["saleDayOfYear"] = df.saledate.dt.dayofyear
    
    df.drop("saledate", axis=1, inplace=True)
    
    # Fill the missing numeric rows with median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing or not
                df[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                df[label] = content.fillna(content.median())
    
        # Fill categorical missing data and turn categories into numbers
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
            # We add +1 to the category code because pandas encodes missing categories as -1
            df[label] = pd.Categorical(content).codes+1
    
    return df

# Process the test data 
df_test = preprocess_data(df_test) #101 collumns
df_test.head()

# We can find how the columns differ using sets
print(set(X_train.columns) - set(df_test.columns))

print(df_test.columns.get_loc('MachineHoursCurrentMeter_is_missing'))

# Manually adjust df_test to have auctioneerID_is_missing column
df_test.insert(56, 'auctioneerID_is_missing', False, True)

xsxs = df_test.columns


#%%

# Make predictions on the test data
test_preds = ideal_model.predict(df_test)

# Format predictions into the same format Kaggle is after
df_preds = pd.DataFrame()
df_preds["SalesID"] = df_test["SalesID"]
df_preds["SalesPrice"] = test_preds

# Export prediction data
df_preds.to_csv('test_predictions.csv', index=False)

#%%
# Find feature importance of our best model
print(ideal_model.feature_importances_)

# Helper function for plotting feature importance
def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features": columns,
                        "feature_importances": importances})
          .sort_values("feature_importances", ascending=False)
          .reset_index(drop=True))
    
    # Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:20])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature importance")
    ax.invert_yaxis()



plot_features(X_train.columns, ideal_model.feature_importances_)















