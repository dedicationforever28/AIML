import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("wind_data.csv", parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# Feature Engineering
df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
df['wind_speed_lag1'] = df['wind_speed'].shift(1)
df['wind_speed_lag2'] = df['wind_speed'].shift(2)

df.dropna(inplace=True)

# Define features and target
X = df[['hour', 'day', 'month', 'wind_speed_lag1', 'wind_speed_lag2']]
y = df['wind_speed']

# Time-based split
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted', linestyle='--')
plt.title("Wind Speed Prediction")
plt.xlabel("Time")
plt.ylabel("Wind Speed")
plt.legend()
plt.show()
