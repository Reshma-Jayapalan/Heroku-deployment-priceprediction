# Importing the libraries
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pickle

#Reading the dataset
dataset = pd.read_csv('housing price.csv')


#The required columns are considered important for prediction (after EDA)
feature_names = ['LotArea','YearBuilt','TotRmsAbvGrd']

#X is assigned dataset with independent variables
X = dataset[feature_names]

#Y is assigned for the dependent variable
y = dataset['SalePrice']

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state=1)

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(X))