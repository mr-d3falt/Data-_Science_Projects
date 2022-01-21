# -*- coding: utf-8 -*-
"""
Model for predicting Heart Dieseases

@author: Araz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_roc_curve 


df =  pd.read_csv('heart-disease.csv')

#Analysing Data

print('\nNumber of missing data :\n',df.isna().sum()) 

print('\n\n\nDataframe info:\n')
print(df.info())

print('\nDescription of Data:\n',df.describe())

df['target'].value_counts().plot(kind='bar',color=['r','b'],)
plt.title("Positive and Negative target Values")
plt.xlabel('1=Disease , 0=No disease')
plt.ylabel('Ammount')
plt.show()

pd.crosstab(df.target,df.sex).plot(kind='bar',figsize=(10,6),color=['r','b'])
plt.title('Heart disease frequency depending on Sex')
plt.xlabel('0=No disease, 1=Disease')
plt.ylabel('Ammount')
plt.legend(['Female','Male'])
plt.xticks(rotation=0)
plt.show()


plt.figure(figsize=(10,6))
#scatter positive examples
plt.scatter(df.age[df.target==1],df.thalach[df.target==1],c='salmon')
#scatter negative examples
plt.scatter(df.age[df.target==0],df.thalach[df.target==0],c='lightblue')

plt.title('Heart Disease in function of Age and Max Heart Rate')
plt.xlabel('Age')
plt.ylabel('Max Heart Rate')
plt.legend(['Disease','No Disease'])
plt.show()


pd.crosstab(df.cp, df.target).plot(kind="bar", figsize=(10, 6),color=["salmon", "lightblue"])

plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Ammount")
plt.legend(["No Disease", "Disease"])
plt.xticks(rotation=0);

#Correlation Matrix to see the dependence of the target on features 
corr_matrix = df.corr()
fig,ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,annot=True,linewidths=0.5,fmt='.2f',cmap='YlGnBu')

#____________________________________________________________________________________________________________________________________
#Selecting a Model

#Split data into X and y

X = df.drop('target',axis=1)
y = df['target']

#Split data into train and test sets

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

models = {'Logistic Regression': LogisticRegression(),
          'KNN':KNeighborsClassifier(),
          'Random Forest':RandomForestClassifier()}

def fit_and_score(models,X_train,X_test,y_train,y_test):
    np.random.seed(42)
    model_scores={}
    
    for name,model in models.items():
        model.fit(X_train,y_train)
        model_scores[name] = model.score(X_test,y_test)
    return  model_scores

model_scores = fit_and_score(models,X_train,X_test,y_train,y_test)

mcmp = pd.DataFrame(model_scores, index = ['accuracy'])
mcmp.T.plot.bar()
plt.xticks(rotation=0)
plt.show()

#Best models are Logistic Regression and Random Forest 

#Tuning Hyperparams using RandomizedSearchCV

# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}


np.random.seed(42)

# Tune LogisticRegression

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(X_train, y_train)
print('\nBest parameters for LogisticRegression:',rs_log_reg.best_params_,'\n\nLogistic Regression score after tuning:',rs_log_reg.score(X_test, y_test))

# Tune  RandomForestClassifier
'''

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(), 
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True,n_jobs=5)

# Fit random hyperparameter search model for RandomForestClassifier()
rs_rf.fit(X_train, y_train)
print('\nBest parameters for  RandomForestClassifier:',rs_rf.best_params_,'\n\n RandomForestClassifier score after tuning:',rs_rf.score(X_test, y_test))
'''

# Different hyperparameters for our LogisticRegression model
log_reg_grid = {"C": np.logspace(-4, 4, 30),
                "solver": ["liblinear"]}

# Setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

# Fit grid hyperparameter search model
gs_log_reg.fit(X_train, y_train);

print('\nBest parameters for LogisticRegression:',gs_log_reg.best_params_,'\n\nLogistic Regression score after tuning:',gs_log_reg.score(X_test, y_test))

#___________________________________________________________________________________________________________________________________
#Model Evaluation

y_preds = gs_log_reg.predict(X_test)

plot_roc_curve(gs_log_reg, X_test, y_test)
plt.show()

sns.set(font_scale=1.5)

def plot_conf_mat(y_test, y_preds):
    """
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.show()
    
    
plot_conf_mat(y_test, y_preds)

print('\n',classification_report(y_test, y_preds))


# Create a new classifier with best parameters
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")

# Cross-validated accuracy
cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="accuracy")
cv_acc = np.mean(cv_acc)

# Cross-validated precision
cv_precision = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="precision")
cv_precision=np.mean(cv_precision)

# Cross-validated recall
cv_recall = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="recall")
cv_recall = np.mean(cv_recall)

# Cross-validated f1-score
cv_f1 = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="f1")
cv_f1 = np.mean(cv_f1)

# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                           "Precision": cv_precision,
                           "Recall": cv_recall,
                           "F1": cv_f1},
                          index=[0])

cv_metrics.T.plot.bar(title="Cross-validated classification metrics",legend=False)
plt.show()

#Feature Importance

# Create a new classifier with best parameters
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")
clf.fit(X_train, y_train);

feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False);





































