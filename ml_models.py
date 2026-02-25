import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Load dataset
df = pd.read_csv("simulation_data.csv")

# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# Features & Target
X = df.drop(columns=["avg_waiting_time"])
y = df["avg_waiting_time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, mse, rmse, r2])

results_df = pd.DataFrame(results, columns=[
    "Model",
    "MAE",
    "MSE",
    "RMSE",
    "R2 Score"
])

results_df = results_df.sort_values(by="R2 Score", ascending=False)
results_df.to_csv("model_comparison.csv", index=False)

print("\n📊 Model Comparison Results:")
print(results_df)

# Feature Importance (for Random Forest)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

importance = pd.Series(rf.feature_importances_, index=X.columns)

plt.figure()
importance.sort_values().plot(kind="barh")
plt.title("Feature Importance - Random Forest")
plt.savefig("feature_importance.png")
plt.close()

print("\n🏆 Best Model:", results_df.iloc[0]["Model"])