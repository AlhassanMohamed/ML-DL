import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('base.csv')

label_encoder = LabelEncoder()
data['context'] = label_encoder.fit_transform(data['context'])

X= data.iloc[:,:-1].values
y= data.iloc[:,-1].values



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=101)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)


# Predicting a new result
y_pred = regressor.predict(np.asarray([[3,1,3,5]]))
y_pred