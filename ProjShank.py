#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 10:36:34 2023

@author: ehaque
may God help any soul that tries to interpret this code 🙏

"""

import numpy as np
import sklearn as sk
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import joblib

#2.1 Data Processing
df = pd.read_csv('Project 1 data.csv')

# test data reading -- print(df.head())

# Setting up chats I think?
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(df['X'], df['Y'], df['Z']);
ax.view_init(0, 0) #plot orient x-y plane about z axis

# correlation heatmap stuff using Seaborn - with the aid of Seaborn

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Pearson Correlation Matrix")
plt.show()

# #redoing dataframe for x y z axis
# dfx = pd.read_csv('Project 1 data.csv', usecols=[0, 0])
# dfy = pd.read_csv('Project 1 data.csv', usecols=[1, 1])
# dfz = pd.read_csv('Project 1 data.csv', usecols=[2, 2])

# # Data for three-dimensional scattered points
# zdata = dfz
# xdata = dfx
# ydata = dfy
# ax.scatter3D(xdata.values, ydata.values, zdata.values);
# '''
# ax.contour3D(xdata~, ydata, zdata, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z');'''

# # sorting by steps 1-13

# # a lil autistic this code is 
# #dfSTEP = pd.read_csv('Project 1 data.csv', usecols = [3,3])

# # if dfSTEP == 1:
#   #  dfS1 = pd.read_csv('Project 1 data.csv', dfSTEP.iloc[0,25])
#     # idk why I went yoda mode fr
    
# # create loop 
    
#     # Read the CSV file with specific columns
# dfSTEP = pd.read_csv('Project 1 data.csv', usecols=[3])

# # Define a condition (e.g., rows where column 1 equals 1)
# #condition = dfSTEP['Step'] == 1

# # Create a new DataFrame containing rows that meet the condition
# #dfS1 = dfSTEP[condition]

# # Create a list to store the DataFrames
# dfS_list = []

# # Create DataFrames for steps 1 to 13
# for step in range(1, 14):
#     condition = dfSTEP['Step'] == step
#     dfS = dfSTEP[condition]
#     dfS_list.append(dfS)