import pickle
import time

import keras.layers
import keras.models
import keras.regularizers
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model
import sklearn.metrics
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import imdb
from keras.models import load_model
from keras.preprocessing.text import Tokenizer

# Load the train and test data
train_data = pd.read_csv("train.csv", header=0) 
test_data = pd.read_csv("test.csv", header=0)
gender_submission = pd.read_csv("gender_submission.csv", header=0)

# Cleaning Data
# drop unnecessary columns
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train_data.drop(drop_cols,axis=1, inplace=True)
test_data.drop(drop_cols,axis=1, inplace=True)

# Adding 

# Fill nan values
train_data["Age"] = train_data["Age"].fillna(train_data.loc[:,"Age"].mean())
test_data["Age"] = train_data["Age"].fillna(train_data.loc[:,"Age"].mean())

train_data = train_data.dropna(subset=['Embarked'])

test_data["Fare"] = test_data["Fare"].fillna(test_data.loc[:,"Fare"].mean())

test_data = test_data.reset_index(drop=True)
train_data = train_data.reset_index(drop=True)

# Create new columns
train_data['Family'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['Family'] = test_data['SibSp'] + test_data['Parch'] + 1

print(train_data["SibSp"].min())
print(train_data["Parch"].min())

interval = (0,130,260,390,520)
categories = ['0-130','130-260','260-390','390-520']
train_data['Fare_Categories'] = pd.cut(train_data.Fare, interval, labels = categories)
test_data['Fare_Categories'] = pd.cut(train_data.Fare, interval, labels = categories)

interval = (0,12,18,30,40,60,122)
categories = ['0-12','12-18','18-30','30-40','40-60','60-122']
train_data['Age_Categories'] = pd.cut(train_data.Age, interval, labels = categories)
test_data['Age_Categories'] = pd.cut(train_data.Age, interval, labels = categories)

interval = (0,3,6,10)
categories = ['0-3','3-6','6-10']
train_data['Family_Size'] = pd.cut(train_data.Family, interval, labels = categories)
test_data['Family_Size'] = pd.cut(train_data.Family, interval, labels = categories)

print(train_data)

# get dataframe info

'''
print("\nTrain Data\n")
print(train_data.info())
print(train_data.head(5))

print("\nTest Data\n")
print(test_data.info())
print(test_data.head(5))
'''

# Ploting each feture against survival rate
'''
sns.catplot(x="Sex", y="Survived", kind="bar", data=train_data)
sns.catplot(x="Pclass", y="Survived", kind="bar", data=train_data)
sns.catplot(x="Embarked", y="Survived", kind="bar", data=train_data)
sns.catplot(x="SibSp", y="Survived", kind="bar", data=train_data)
sns.catplot(x="Family", y="Survived", kind="bar", data=train_data)
sns.catplot(x="Parch", y="Survived", kind="bar", data=train_data)
sns.catplot(x="Age_Categories", y="Survived", hue="Sex", kind="bar", data=train_data)
sns.catplot(x="Fare_Categories", y="Survived", kind="bar", data=train_data)
plt.show()
'''

# Separate labels from the rest of the input features
train_labels = train_data["Survived"]
train_data = train_data.drop(columns="Survived")
test_labels = gender_submission["Survived"]

# Encode all categorical variables
train_data = pd.get_dummies(train_data, prefix_sep="_", drop_first=True)
test_data = pd.get_dummies(test_data, prefix_sep="_", drop_first=True)

# Select columns of interest
cols = train_data.columns

# get dataframe info
'''
print("\nTrain Data\n")
print(train_data.info())
print(train_data.head(5))

print("\nTest Data\n")
print(test_data.info())
print(test_data.head(5))
'''

# One-hot encode the labels
train_labels_onehot = keras.utils.to_categorical(train_labels)
test_labels_onehot = keras.utils.to_categorical(test_labels)
num_classes = test_labels_onehot.shape[1]

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

model = load_model('best_model.h5')
print(model.predict_classes(test_data, verbose=1))


results2 = pd.read_csv("test.csv", header=0)

results2['Survived'] = model.predict_classes(test_data, verbose=1)
cols = ['PassengerId', 'Survived']
results2 = results2[cols]
results2 = results2.set_index('PassengerId')
print(results2.columns)
print(results2)

try:
    file_name = 'CNN Results.csv'
    results2.to_csv(file_name)
except:
    print("An exception occurred")
else:
    print("File", file_name, "saved!")
