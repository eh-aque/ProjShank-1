import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

"""
Code wasn't predicting properly, Shashank told me to remove this section'
# Feature scaling
scaler = StandardScaler()
coords_train = scaler.fit_transform(coords_train)
coords_test = scaler.transform(coords_test)

"""

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
print("Accuracy:", accuracy_score(step_test, svm_predictions))
print("Confusion Matrix:\n", confusion_matrix(step_test, svm_predictions))
print("Classification Report:\n", classification_report(step_test, svm_predictions))

# Model 3: Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(coords_train, step_train)
rf_predictions = rf_classifier.predict(coords_test)

print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(step_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(step_test, rf_predictions))
print("Classification Report:\n", classification_report(step_test, rf_predictions))

# Save and load the selected model (Model 3: Random Forest)
joblib.dump(rf_classifier, 'rf_model.joblib')
loaded_rf_model = joblib.load('rf_model.joblib')

# Coordinates for prediction
coordinates_to_predict = np.array([[9.375, 3.0625, 1.51],
                                   [6.995, 5.125, 0.3875],
                                   [0, 3.0625, 1.93],
                                   [9.4, 3, 1.8],
                                   [9.4, 3, 1.3]])

# Make predictions using the loaded model for each set of coordinates
for idx, coord_set in enumerate(coordinates_to_predict):
    step_prediction = loaded_rf_model.predict(coord_set.reshape(1, -1))
    print(f"Predicted Step for coordinates {idx + 1}: {step_prediction}")
    
    
    # use to remove warnings
# logistic_regressor = LogisticRegression(max_iter=1000)
# classification_report(step_test, logistic_predictions, zero_division=1)
# feature_names = ['X', 'Y', 'Z']
# rf_classifier = RandomForestClassifier(feature_names=feature_names)


