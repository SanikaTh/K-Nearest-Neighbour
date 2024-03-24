# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:12:31 2024

@author: HP
"""

'''1.	A glass manufacturing plant uses different earth elements to design new
 glass materials based on customer requirements. For that, they would like to automate
 the process of classification as it’s a tedious job to manually classify them. Help
 the company achieve its objective by correctly classifying the glass type based on the
 other features using KNN algorithm. 
 
 The objective of implementing the K-nearest neighbors algorithm for classification of glasses
 based on customer requirements based on their attributes. It includes the accuracy and
 efficiency. 
 the constraints are performance of KNN model is evaluated through the accuracy,
 recall, F1 score and prediction for classification of glasses'''

#Load the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the glass dataset (replace 'glass.csv' with your actual dataset file)
glass_data = pd.read_csv("D:/Documents/Datasets/Glass.csv")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(glass_data.head())
print(glass_data.info())
print(glass_data.describe())
# Separate features (X) and target variable (y)
X = glass_data.drop('Type', axis=1)  # Features
y = glass_data['Type']  # Target variable
3
#normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#let us now apply this function to dataframe
glass_data_n=norm_func(glass_data.iloc[:,1:32])
# because now 0th column is output or label it is not considered 

################################
#let us now apply X as input and y as output

X=np.array(glass_data_n.iloc[:,:])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features by scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed

# Train the KNN classifier
knn.fit(X_train_scaled, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
# 0.7674418604651163

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

'''
precision    recall  f1-score   support

           1       0.73      1.00      0.85        11
           2       0.75      0.64      0.69        14
           3       1.00      0.33      0.50         3
           5       0.67      0.50      0.57         4
           6       0.75      1.00      0.86         3
           7       0.88      0.88      0.88         8

    accuracy                           0.77        43
   macro avg       0.80      0.73      0.72        43
weighted avg       0.78      0.77      0.75        43

'''

