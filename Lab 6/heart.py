import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
heart = pd.read_csv("heart.csv")

print(heart.head())
print(heart.describe())

# -------- EDA --------

plt.figure(figsize=(6,4))
sns.heatmap(heart.corr(), annot=True)
plt.title("Heart Dataset Correlation")
plt.show()

sns.countplot(x="target", data=heart)
plt.title("Heart Disease Distribution")
plt.show()

# -------- Model --------

X = heart.drop("target",axis=1)
y = heart["target"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Find best K value
accuracy_rates = []

for k in range(1,21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    accuracy_rates.append(accuracy_score(y_test,pred))

plt.plot(range(1,21),accuracy_rates)
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("Choosing Optimal K")
plt.show()

# Best model
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

print("\nClassification Report")
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True,fmt="d",cmap="Reds")
plt.title("Confusion Matrix - Heart Dataset")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
