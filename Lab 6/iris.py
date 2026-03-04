import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = pd.read_csv("iris (1).csv")

print(iris.head())
print(iris.describe())

# -------- EDA --------

sns.pairplot(iris, hue="species")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(iris.drop("species",axis=1).corr(), annot=True)
plt.title("Iris Correlation Heatmap")
plt.show()

# -------- Model --------

X = iris.drop("species", axis=1)
y = iris["species"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# choose K value
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,y_pred))

print("\nClassification Report")
print(classification_report(y_test,y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.title("Confusion Matrix - Iris")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
