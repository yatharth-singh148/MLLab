import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
diabetes = pd.read_csv("diabetes.csv")

print(diabetes.head())
print(diabetes.describe())

# -------- EDA --------

plt.figure(figsize=(6,4))
sns.heatmap(diabetes.corr(), annot=True)
plt.title("Diabetes Correlation Heatmap")
plt.show()

sns.countplot(x="Outcome", data=diabetes)
plt.title("Diabetes Outcome Distribution")
plt.show()

# -------- Model --------

X = diabetes.drop("Outcome",axis=1)
y = diabetes["Outcome"]

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True,fmt="d",cmap="Greens")
plt.title("Confusion Matrix - Diabetes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
