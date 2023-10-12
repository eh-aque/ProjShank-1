# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Sep 28 10:36:34 2023

# @author: ehaque
# may God help any soul that tries to interpret this code 🙏

# """

# import numpy as np
# import sklearn as sk
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# #2.1 Data Processing
# df = pd.read_csv('Project 1 data.csv')

# # test data reading -- print(df.head())

# # Setting up chats I think?
# #2.2 Data Visualization
# from mpl_toolkits import mplot3d

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# ax.scatter3D(df['X'], df['Y'], df['Z']);
# ax.view_init(0, 0) #plot orient x-y plane about z axis

# # correlation heatmap stuff using Seaborn - with the aid of Seaborn
# #2.3 Correlation Analysis
# correlation_matrix = df.corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='PiYG', fmt=".2f", linewidths=0.5)
# plt.title("Pearson Correlation Matrix")
# plt.show()

# #2.4 Classification Model Dev
# # get coordinates output step, therefore model should have same format
# # split data frame into coord & step (2 frames liek 2 chainz)
# # Split the DataFrame by columns
# dfcoords = df[['X', 'Y', 'Z']]  # Create a DataFrame with columns X Y Z
# dfstep = df[['Step']]  # Create a DataFrame with column Step

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# # xyz input steps is output therefore train xyz to predict step

# # Split the data into training and testing sets
# # X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(dfcoords[['X']], dfcoords[['Y']], dfcoords[['Z']], test_size=0.2, random_state=42)
# coords_train, coords_test = train_test_split(dfcoords, test_size=0.2, random_state=42)

# #feature scaling
# scaler = StandardScaler()
# coords_train = scaler.fit_transform(coords_train)
# coords_test = scaler.transform(coords_test)

# #classifier 
# classifier = LogisticRegression()

# classifier.fit(coords_train, coords_train)

# step_pred = classifier.predict(coords_test)

# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# accuracy = accuracy_score(coords_test, step_pred)
# confusion = confusion_matrix(coords_test, step_pred)
# report = classification_report(coords_test, step_pred)

# print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", confusion)
# print("Classification Report:\n", report)


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

# Create a Linear Regression model
regressor = LinearRegression()

# Train the model
regressor.fit(coords_train, step_train)

# Make predictions
step_pred = regressor.predict(coords_test)

# Evaluate the model's performance
mse = mean_squared_error(step_test, step_pred)
r2 = r2_score(step_test, step_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)






























