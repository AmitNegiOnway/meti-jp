import pandas as pd
import numpy as np
import pickle
import yaml
from xgboost import XGBRegressor

# Load data
X_train = pd.read_csv('data/interim/X_train_scaled.csv')
y_train = pd.read_csv('data/interim/y_train.csv')

X_train = X_train.values
y_train = y_train.values.flatten()

# Load fixed params from YAML
with open("params.yaml") as f:
    params = yaml.safe_load(f)["model_building"]

# Train manually using your chosen params
xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_estimators=params['n_estimators'],
    learning_rate=params['learning_rate'],
    max_depth=params['max_depth'],
    subsample=params['subsample'],
    colsample_bytree=params['colsample_bytree']
)

xgb.fit(X_train, y_train)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(xgb, f)
