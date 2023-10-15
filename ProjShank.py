import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

# Load the dataset
df = pd.read_csv('Project 1 data.csv')

# Data Visualization
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['X'], df['Y'], df['Z'])
ax.view_init(30, 185)  # Set the view angle

plt.show()

# Correlation Analysis
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='PiYG', fmt=".2f", linewidths=0.5)
plt.title("Pearson Correlation Matrix")
plt.show()

# Classification Model Development
# Separate features (X: 'X', 'Y', 'Z') and target (y: 'Step')
coords = df[['X', 'Y', 'Z']]
step = df['Step']

# Split the data into training and testing sets
coords_train, coords_test, step_train, step_test = train_test_split(coords, step, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
coords_train = scaler.fit_transform(coords_train)
coords_test = scaler.transform(coords_test)

# Model 1: Logistic Regression
logistic_regressor = LogisticRegression()
logistic_regressor.fit(coords_train, step_train)
logistic_predictions = logistic_regressor.predict(coords_test)

print("Logistic Regression:")
print("Accuracy:", accuracy_score(step_test, logistic_predictions))
print("Confusion Matrix:\n", confusion_matrix(step_test, logistic_predictions))
print("Classification Report:\n", classification_report(step_test, logistic_predictions))

# Model 2: Support Vector Machine (SVM)
svm_classifier = SVC()
svm_classifier.fit(coords_train, step_train)
svm_predictions = svm_classifier.predict(coords_test)

print("Support Vector Machine:")
print("Accuracy:", accuracy_score(step_test,svm_predictions))  # This should be replaced with appropriate regression metric
print("Confusion Matrix:\n", confusion_matrix(step_test, svm_predictions))  # This should be removed for regression
print("Classification Report:\n", classification_report(step_test,svm_predictions))  # This should be removed for regression

# Model 3: Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(coords_train, step_train)
rf_predictions = rf_classifier.predict(coords_test)

print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(step_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(step_test, rf_predictions))
print("Classification Report:\n", classification_report(step_test, rf_predictions))  # This should be removed for regression


"""Save and load selected model, model 2, SVM"""

joblib.dump(svm_classifier, 'svm_model.joblib')
loaded_svm_model = joblib.load('svm_model.joblib')

coordinates_to_predict = np.array([[9.375, 3.0625, 1.51],
                                   [6.995, 5.125, 0.3875],
                                   [0, 3.0625, 1.93],
                                   [9.4, 3, 1.8],
                                   [9.4, 3, 1.3]])

for idx, coord_set in enumerate(coordinates_to_predict):
    step_prediction = loaded_svm_model.predict(coord_set.reshape(1, -1))
    print(f"Predicted Step for coordinates {idx + 1}: {step_prediction}")



