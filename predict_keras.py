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

'''

Data Dictionary
Variable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex	
Age	Age in years	
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

'''

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

# Training constants
n_layers = 3
n_nodes = 500
batch_size = 32
n_epochs = 1000
dropout_rate = 0.60

n_nodes_per_layer = n_nodes // n_layers
n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {}".format(batch_size, n_batches, n_epochs))
print("Num layers: {}  total num nodes: {}".format(n_layers, n_nodes))
print("Dropout rate: {}".format(dropout_rate))

#
# Keras definitions
#

# Create a neural network model
model = keras.models.Sequential()

# First layer (need to specify the input size)
print("Adding layer with {} nodes".format(n_nodes_per_layer))
model.add(keras.layers.Dense( n_nodes_per_layer, input_shape=(n_inputs,),
        activation='elu',
        kernel_initializer='he_normal', bias_initializer='zeros'))
model.add(keras.layers.Dropout(dropout_rate))

# Other hidden layers
for n in range(1, n_layers):
    print("Adding layer with {} nodes".format(n_nodes_per_layer))
    model.add(keras.layers.Dense( n_nodes_per_layer, 
        activation='elu',
        kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(keras.layers.Dropout(dropout_rate))

# Output layer
print("Adding layer with {} nodes".format(num_classes))
model.add(keras.layers.Dense(num_classes,
        activation='softmax',
        kernel_initializer='glorot_normal', bias_initializer='zeros'))

# Define the optimizer
#optimizer = keras.optimizers.Adam(lr=0.001)
optimizer = keras.optimizers.Adadelta(lr=1.0)
#optimizer = keras.optimizers.Adagrad(lr=0.01)
        
# Define cost function and optimization strategy
model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

# Set callback functions to early stop training and save the best model so far
callback = [EarlyStopping(monitor='val_loss', patience=200, verbose=1),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

# Train the neural network
start_time = time.time()
history = model.fit(
        train_data,
        train_labels_onehot,
        epochs=n_epochs,
        callbacks = callback,
        batch_size=batch_size,
        validation_data=(test_data, test_labels_onehot),
        verbose=2,
        )

end_time = time.time()
print("Training time: ", end_time - start_time);

# Find the best costs & metrics
test_accuracy_hist = history.history['val_acc']
best_idx = test_accuracy_hist.index(max(test_accuracy_hist))
print("Max test accuracy:  {:.4f} at epoch: {}".format(test_accuracy_hist[best_idx], best_idx))

trn_accuracy_hist = history.history['acc']
best_idx = trn_accuracy_hist.index(max(trn_accuracy_hist))
print("Max train accuracy: {:.4f} at epoch: {}".format(trn_accuracy_hist[best_idx], best_idx))

test_cost_hist = history.history['val_loss']
best_idx = test_cost_hist.index(min(test_cost_hist))
print("Min test cost:  {:.5f} at epoch: {}".format(test_cost_hist[best_idx], best_idx))

trn_cost_hist = history.history['loss']
best_idx = trn_cost_hist.index(min(trn_cost_hist))
print("Min train cost: {:.5f} at epoch: {}".format(trn_cost_hist[best_idx], best_idx))

# Plot the history of the cost
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')

# Plot the history of the metric
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
