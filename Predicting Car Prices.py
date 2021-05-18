# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:01:11 2021

@author: ananya Joshi
"""

'''
In this project we will predict the car prices using it's attributes by K Nearest Neighbors 
algorithm based on Regression'

For each car we have info. about technical aspects such as motor's displacement, weight of car, 
miles per gallon and so on.
'''

'''
#--To read more about dataset uncomment this section and execute code to go to dataset page--
#-- dataset documentation --

import webbrowser
webbrowser.open('https://archive.ics.uci.edu/ml/datasets/automobile')
'''


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



pd.set_option('display.max_columns',26)
# using column names from the dataset documentation mentioned above
cols = ['symbolizing','normalized-losses','make','fuel-type','aspiration','num-of-doors',
        'body-style','drive-wheels','engine-location','wheel-base','length','width','height',
        'curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore',
        'stroke', 'compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg',
        'price']
cars = pd.read_csv('E:\DataQuest\Projects\K Nearest Neighbors\imports-85.data',names=cols)

print('---------------------------------')
# determining the column types in dataset
print(cars.info())
print('---------------------------------')
print(cars.isnull().sum())

# let's see few starting rows of dataframe
print('---------------------------------')
print(cars.head(2))

'''
After seeing the data we can clearly see our target column i.e. price which we will predict.
Since we are using KNN based on regression we need to avoid the features that will affect
our models performance and in case of KNN based on regression

features that:
    i. have non-numeric values
    ii. have numeric values but are not ordinal
    iii. geographical values
such features are of no use in this algorithm therefore we can say that it's safe to remove
these columns.

numeric columns in our data

-> symbolizing, wheel-base, length, width, height, curb-weight, engine-size,compression-ratio,
city-mpg, highway-mpg

all other columns are not numeric even out target column i.e. price is not numeric
Also if we are making a predicting model then our data must not have any missing values.

This means that our data here now need some cleaning

'''

'''
our normalized-losses column contains missing values as '?' So, let us replace all ? with
np.nan in whole dataset. So, whereever there is '?' it will be rekpllaced by numpy.nan
'''
print('---------------------------------')
cars = cars.replace('?',np.nan)
# checking the null values in dataset again
print(cars.isnull().sum())
# selecting numeric columns only
numeric_cols = cols = ['symbolizing','normalized-losses','wheel-base','length','width','height'
,'curb-weight','engine-size','bore','stroke', 'compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg',
'price']
cars_num_df = cars[numeric_cols]

print('---------------------------------')
# Now we have all numeric data in our data thus we can convert it all to numeric by astype
cars_num_df = cars_num_df.astype('float')
print(cars_num_df.dtypes)

'''
Now, we have all numeric data in our dataframe and our data is cleaned in a way to predict
pricing of cars
'''
print('---------------------------------')
print(cars_num_df.shape) #So, we have ony 205 rows
print('Null values in normalized-losses : ',cars_num_df['normalized-losses'].isnull().sum())
'''
means that out of 205 rows in normalized-losses feature there are 41 missing values so, now
we have 3 options to opt from:
    i. replace missing values with average value of column or,
    ii. drop the rows entirely or,
    iii. drop the column entirely
    
Since, here there are about 25% approx. of rows are missing values so it's better to 
drop the entire column as dropping rows will affect other columns and result in overall
less data for our model.

More the 'correct' data the better it is for model
'''
plt.figure(figsize = (20,20), dpi=400)
sns.heatmap(cars_num_df, cbar=False)
plt.title('Before Removing Missing values')
plt.show()



# dropping normalized-losses column
cars_num_df.drop(['normalized-losses'],inplace=True,axis=1)
cars_num_df.columns
print('---------------------------------')
# let us check out other columns also for missing values and take appropriate action
print(cars_num_df.isnull().sum())

'''
So, we have null values in following columns:
    bore
    stroke
    price

Now, worst case to assume is that all these missing values rows are different.
Since, in total we have less rows it's better to replace them with average value of column
'''

for col in cars_num_df:
    cars_num_df[col].fillna(value=cars_num_df[col].mean(),inplace=True)

print('---------------------------------')
print(cars_num_df.isnull().sum())
print('---------------------------------')

'''
Since, we selected above columns to be in our model so, now let us move forward
'''
print('---------------------------------')
# Now, let us normalize our data
normalized_cars_df = (cars_num_df - cars_num_df.mean())/cars_num_df.std()

print(normalized_cars_df.isnull().sum())
plt.figure(figsize = (20,20), dpi=400)
sns.heatmap(normalized_cars_df.isnull(), cbar=False)
plt.title('After Removing Missing Values')
plt.show()
print('---------------------------------')


# UNIVARIATE MODEL

# using simple train/test validation
# print(normalized_cars_df.shape)

univariate_rmses = []
def knn_train_test(df, target_col, feature, k):
    train_df = df.iloc[:154]
    test_df = df.iloc[154:]
    model = KNeighborsRegressor(algorithm = 'brute', n_neighbors = k)
    model.fit(X = train_df[[feature]], y = train_df[target_col])
    predictions = model.predict(X = test_df[[feature]])
    rmse = np.sqrt(mean_squared_error(y_true = test_df['price'], y_pred = predictions))
    univariate_rmses.append(rmse)
    
'''
Now, using each column one by one we will compute the rmse value for our
univariate model. We will not use 'price' column as using it in case of prediction
model leads towards data-leakage in machine learning as it is our target column.
'''
univariate_columns = [x for x in normalized_cars_df.columns if x != 'price']

# calling function to train model

'''
# Uncomment this section to get rmse value for each feature
for col in univariate_columns:
    knn_train_test(normalized_cars_df, 'price', col)
'''

# for now we will use below mentioned columns
# cols -> curb-weight, highway-mpg, city-mpg, width, engine-size
univariate_features = 'curb-weight'
hyper_params = [1, 3, 5, 7, 9]


for k in hyper_params:
    knn_train_test(normalized_cars_df, 'price', col, k)

print('RMSE values are : ',univariate_rmses)
print('---------------------------------')

print('Univariate Hyper-parameter values : ')
print(hyper_params,'\n')
print('Univariate-model RMSE values for above hyper-parameter values')
print(univariate_rmses)
import matplotlib.pyplot as plt
plt.scatter(x = hyper_params, y = univariate_rmses)
plt.xlabel('Hyper-Parameter')
plt.ylabel('Root Mean Squared Error')
plt.title('Univariate RMSES')
plt.show()

'''
So, based on the scatter-plot we have the lowest rmse value at k = 5
This lowest rmse computed value is for a univariate model with feature -> 'curb-weight'
'''

# MULTIVARIATE MODEL
print('---------------------------------')
'''
Building a good model:
    i. changing features
    ii. hyperparameter optimization
'''
np.random.seed(1)
shuffled_index = np.random.permutation(len(normalized_cars_df))
normalized_cars_df = normalized_cars_df.iloc[shuffled_index]
def knn_train_test_mult(df, target_col, features_list):
    mult_rmses = []
    train_df = df.iloc[:154]
    test_df = df.iloc[154:]
    #for feature in features_list:
    model = KNeighborsRegressor(algorithm = 'brute', n_neighbors = 5)
    model.fit(X = train_df[features_list], y = train_df[target_col])
    predictions = model.predict(X = test_df[features_list])
    rmse = np.sqrt(mean_squared_error(y_true = test_df['price'], y_pred = predictions))
    mult_rmses.append(rmse)
    return mult_rmses
# calling function for multivariate


best_5_features_list = ['curb-weight','engine-size','highway-mpg','width','city-mpg']
two_features = knn_train_test_mult(normalized_cars_df, 'price', best_5_features_list[:2])
three_features = knn_train_test_mult(normalized_cars_df, 'price', best_5_features_list[:3])
four_features = knn_train_test_mult(normalized_cars_df, 'price', best_5_features_list[:4])
five_features = knn_train_test_mult(normalized_cars_df, 'price', best_5_features_list[:5])

models_avg_rmse_dict = {'2-features model\'s RMSE':two_features[0], '3-features model\'s RMSE':three_features[0],'4-features model\'s RMSE':four_features[0],'5-features model\'s RMSE':five_features[0]}


# displaying all rmse values computed from above multivariate model
print('Multivariate model RMSE values for model of 2,3,4,5 features respectively : \n')
print (' two_features_rmse   -> {}\n three_features_rmse -> {}\n four features_rmse  -> {}\n five_features_rmse  -> {}'.format(two_features,three_features,four_features,five_features))
print('\nOR\n')
print(models_avg_rmse_dict)
plt.scatter(x = [5,5,5,5], y = [two_features,three_features,four_features,five_features])
plt.title('Multivariate-Model\'s RMSE Compared')
plt.xlabel('Hyper-Parameter')
plt.ylabel('Root Mean Squared Value')
plt.ylim(0.35,0.55)
plt.show()
'''
So, from above scatterplot we can clearly see that Minimum RMSE value of 0.4435 approx. when 
we used only all of the 5 features
This proves that using all these 5 features the model is performing good
'''

# PERFORMING HYPERPARAMETER TUNING
'''
Top 3 top models selected (based on RMSE values):
    i. four_features model (RMSE = 0.4435)
    ii. three_features model (RMSE = 0.4631)
    iii. five_features model (RMSE = 0.4462)
'''
hyper_param_opt = [x for x in range(1,26,1)]
rmses_3_features = []
rmses_4_features = []
rmses_5_features = []
# just rewriting knn_train_test_mult
def knn_train_test_mult(df, target_col, features_list,hyper_param_opt,n_features):
    train_df = df.iloc[:154]
    test_df = df.iloc[154:]
    #for feature in features_list:
    for hp in hyper_param_opt:
        model = KNeighborsRegressor(algorithm = 'brute', n_neighbors = hp)
        model.fit(X = train_df[features_list], y = train_df[target_col])
        predictions = model.predict(X = test_df[features_list])
        rmse = np.sqrt(mean_squared_error(y_true = test_df['price'], y_pred = predictions))
        if n_features == 3:
            rmses_3_features.append(rmse)
        elif n_features == 4:
            rmses_4_features.append(rmse)
        elif n_features == 5:
            rmses_5_features.append(rmse)
    return predictions


predictions_3_features = knn_train_test_mult(normalized_cars_df, 'price', best_5_features_list[:3], hyper_param_opt, 3)
predictions_4_features = knn_train_test_mult(normalized_cars_df, 'price', best_5_features_list[:4], hyper_param_opt, 4)
predictions_5_features = knn_train_test_mult(normalized_cars_df, 'price', best_5_features_list[:5], hyper_param_opt, 5)

fig = plt.figure(figsize=(20,20),dpi=300)

ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

ax1.scatter(x=hyper_param_opt,y=rmses_3_features,c='black')
ax1.set_title('3-features model\'s RMSES')
ax1.set_ylabel('RMSE')
ax1.set_xlabel('Hyperparameter')
ax1.set_facecolor('orange')
ax2.scatter(x=hyper_param_opt,y=rmses_4_features,c='black')
ax2.set_ylabel('RMSE')
ax2.set_xlabel('Hyperparameter')
ax2.set_facecolor('orange')
ax2.set_title('4-features model\'s RMSES')
ax3.scatter(x=hyper_param_opt,y=rmses_5_features,c='black')
ax3.set_ylabel('RMSE')
ax3.set_facecolor('orange')
ax3.set_xlabel('Hyperparameter')
ax3.set_title('5-features model\'s RMSES')
fig.suptitle("Top 3-Models RMSES's with Varying Hyperparameters")
plt.show()

'''
In 3-feature model:
    Minimum RMSE is at k-value of 3 so k=3 is our best build model
In 4-feature model:
    Minimum RMSE is at k-value of 3 so k=3 is our best build model
In 5-feature model:
    Minimum RMSE is at k-value of 2 so k=2 is our best build model
'''

# Performing K-Fold Cross-Validation
'''
To accurately understand the model's performance:
    i. Perform K-Fold Cross-Validation
    (Since we have less data we will stick with default k value i.e. 5)
And,
Using k-hyperparameter as respective value mentioned above
'''

print('---------------------------------\n')

fold = 5
# for single model only
# instantiating KFold class
num_folds = [3, 5, 10, 15]
# df = cars_num_df
def kfold_cv(df, target_col, features_list, n_features):
    print('\n\n{}-FEATURE MODEL:\n'.format(n_features))
    for n_fold in num_folds:
        kf = KFold(n_splits = n_fold, shuffle = True, random_state = 1)
        # instantiating KNN class
        knn = KNeighborsRegressor(algorithm = 'brute', n_neighbors=5)
        # computing score metric
        mse = cross_val_score(X = df[features_list], estimator = knn, y = df[target_col], cv = kf, scoring = 'neg_mean_squared_error')
        rmses = np.sqrt(np.abs(mse))
        avg_rmse = np.mean(rmses)
        std_rmse = np.std(rmses)
        print('{}-feature model\'s avg-RMSE is {} and std-RMSE is {}'.format(n_features,avg_rmse,std_rmse))
    
    

# calling 3 feature model
kfold_cv(cars_num_df,'price',best_5_features_list[:3],3)
kfold_cv(cars_num_df,'price',best_5_features_list[:4],4)
kfold_cv(cars_num_df,'price',best_5_features_list[:5],5)


