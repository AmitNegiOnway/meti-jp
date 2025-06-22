import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

clf = pickle.load(open('models/model.pkl', 'rb'))
X_test = pd.read_csv('data/interim/X_test_scaled.csv')
y_test = pd.read_csv('data/interim/y_test.csv')

X_test = X_test.values
y_test = y_test.values

y_pred = clf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metric_dict = {
    'mse': mse,
    'mae': mae,
    'r2': r2
}

with open('metrics.json', 'w') as file:
    json.dump(metric_dict, file, indent=4)
