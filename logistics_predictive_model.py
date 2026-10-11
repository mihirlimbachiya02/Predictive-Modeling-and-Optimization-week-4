logistics_predictive_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

# 1. Simulate Logistics Prediction Dataset
np.random.seed(42)
n_samples = 2000

data = {
    "distance_km": np.random.uniform(1.5, 40.0, n_samples),
    "package_weight_kg": np.random.exponential(scale=3.5, size=n_samples),
    "traffic_density_index": np.random.randint(1, 10, size=n_samples),
    "weather_condition_encoded": np.random.randint(
        0, 3, size=n_samples
    ),  # 0: Clear, 1: Rain, 2: Fog
    "driver_experience_years": np.random.randint(0, 10, size=n_samples),
}
df = pd.DataFrame(data)

# Simulate target variable: delivery_time_mins
df["delivery_time_mins"] = (
    8
    + (df["distance_km"] * 1.3)
    + (df["traffic_density_index"] * 2.8)
    + (df["weather_condition_encoded"] * 4.0)
    - (df["driver_experience_years"] * 0.5)
    + np.random.normal(0, 2.5, n_samples)
)

# 2. Features and Target Splitting
X = df.drop(columns=["delivery_time_mins"])
y = df["delivery_time_mins"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Model Training & Hyperparameter Tuning (Grid Search for Gradient Boosting)
param_grid = {
    "n_estimators": [50, 100],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
}
gb_model = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# 4. Evaluation and Metrics
predictions = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Model RMSE: {rmse:.2f} mins")
print(f"Model MAE: {mae:.2f} mins")
print(f"Model R-Squared (R2): {r2:.4f}")
