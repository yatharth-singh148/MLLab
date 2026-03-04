import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
petrol = pd.read_csv("petrol_consumption.csv")

print(petrol.head())
print(petrol.describe())

# ---------------- EDA ----------------

plt.figure(figsize=(6,4))
sns.heatmap(petrol.corr(), annot=True)
plt.title("Petrol Dataset Correlation")
plt.show()

sns.pairplot(petrol)
plt.show()

# ---------------- Model ----------------

X = petrol.drop("Petrol_Consumption",axis=1)
y = petrol["Petrol_Consumption"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeRegressor()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# ---------------- Evaluation ----------------

print("MAE:",mean_absolute_error(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred)))

# Plot predictions
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Petrol Consumption")
plt.ylabel("Predicted Petrol Consumption")
plt.title("Actual vs Predicted")
plt.show()
