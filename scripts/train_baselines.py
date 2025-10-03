
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

data = np.load('data/processed/train.npz')
X_train, y_train = data['X'], data['y']
data = np.load('data/processed/test.npz')
X_test, y_test = data['X'], data['y']

rf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
prob = rf.predict_proba(X_test)[:,1]
pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1:", f1_score(y_test, pred))
print("AUC:", roc_auc_score(y_test, prob))

pd.DataFrame(confusion_matrix(y_test, pred)).to_csv("outputs/confusion_rf.csv")
