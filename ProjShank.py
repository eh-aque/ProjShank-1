import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
#from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('Project 1 data.csv')


#2.2 Data Visualization
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(df['X'], df['Y'], df['Z']);
ax.view_init(0, 0) #plot orient x-y plane about z axis


# correlation heatmap stuff using Seaborn - with the aid of Seaborn
#2.3 Correlation Analysis
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='PiYG', fmt=".2f", linewidths=0.5)
plt.title("Pearson Correlation Matrix")
plt.show()


#2.4 Classification Model Dev
# Separate features (X: 'X', 'Y', 'Z') and target (y: 'Step')
coords = df[['X', 'Y', 'Z']]
step = df['Step']

# Split the data into training and testing sets
coords_train, coords_test, step_train, step_test = train_test_split(coords, step, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
coords_train = scaler.fit_transform(coords_train)
coords_test = scaler.transform(coords_test)



'''model 1 Linear Regression'''

# Create a Linear Regression model
"""11111111111111111111"""
linear_regressor = LinearRegression()

# Train the model
linear_regressor.fit(coords_train, step_train)

# Make predictions
linear_predictions = linear_regressor.predict(coords_test)

# Evaluate the model's performance

#mse = mean_squared_error(step_test, step_pred)
#r2 = r2_score(step_test, step_pred)
#print("Mean Squared Error:", mse)
#print("R-squared:", r2)

print("Linear Regression:")
print("Accuracy:", accuracy_score(step_test,linear_predictions))
print("Confusion Matrix:\n", confusion_matrix(step_test, linear_predictions))
print("Classification Report:\n", classification_report(step_test,linear_predictions))



'''model 2 SVM'''

# Create a Support Vector Machine model
"""22222222222222222222"""
svm_classifier = SVC()

# Train the model
svm_classifier.fit(coords_train, step_train)

# Make predictions
svm_predictions = svm_classifier.predict(coords_test)

# Evaluate the model's performance

#mse = mean_squared_error(step_test, step_pred)
#r2 = r2_score(step_test, step_pred)
#print("Mean Squared Error:", mse)
#print("R-squared:", r2)

print("Support Vector Machine:")
print("Accuracy:", accuracy_score(step_test,svm_predictions))
print("Confusion Matrix:\n", confusion_matrix(step_test, svm_predictions))
print("Classification Report:\n", classification_report(step_test,svm_predictions))



'''model 3 Random Forrest'''

# Create a Random Forrest model
"""*3333333333333333333"""
rf_classifier = RandomForestClassifier()

# Train the model
rf_classifier.fit(coords_train, step_train)

# Make predictions
rf_predictions = rf_classifier.predict(coords_test)

# Evaluate the model's performance

#mse = mean_squared_error(step_test, step_pred)
#r2 = r2_score(step_test, step_pred)
#print("Mean Squared Error:", mse)
#print("R-squared:", r2)

print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(step_test,rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(step_test, rf_predictions))
print("Classification Report:\n", classification_report(step_test,rf_predictions))

























