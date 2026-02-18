import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


# Load dataset
data = pd.read_csv("dataset.csv")

X = data[["size"]]
y = data["price"]

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
print("R2 Score:", r2_score(y_test, predictions))
