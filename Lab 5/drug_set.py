import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
drug = pd.read_csv("drug.csv")

print(drug.head())

# ---------------- EDA ----------------

sns.countplot(x="Drug", data=drug)
plt.title("Drug Distribution")
plt.show()

sns.boxplot(x="Drug", y="Age", data=drug)
plt.title("Age vs Drug")
plt.show()

# ---------------- Encoding ----------------

le = LabelEncoder()

for col in drug.columns:
    drug[col] = le.fit_transform(drug[col])

# ---------------- Model ----------------

X = drug.drop("Drug",axis=1)
y = drug["Drug"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier(criterion="entropy")

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

# Confusion matrix
cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True,fmt="d")
plt.title("Confusion Matrix - Drug")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
