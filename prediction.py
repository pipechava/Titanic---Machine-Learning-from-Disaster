import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns

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
train_data['Family'] = train_data['SibSp'] + train_data['Parch']
test_data['Family'] = test_data['SibSp'] + test_data['Parch']

interval = (0,130,260,390,520)
categories = ['0-130','130-260','260-390','390-520']
train_data['Fare_Categories'] = pd.cut(train_data.Fare, interval, labels = categories)
test_data['Fare_Categories'] = pd.cut(train_data.Fare, interval, labels = categories)

interval = (0,12,18,30,40,60,122)
categories = ['0-12','12-18','18-30','30-40','40-60','60-122']
train_data['Age_Categories'] = pd.cut(train_data.Age, interval, labels = categories)
test_data['Age_Categories'] = pd.cut(train_data.Age, interval, labels = categories)


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

##############################
#Train

############################## Binary Classification with Linear Regression ##############################
# Create and train a new LinearRegression model
model = sklearn.linear_model.LinearRegression()

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Predict new labels for test data
Y_pred_proba = model.predict(test_data[cols])

# Binarize the predictions by comparing to a threshold
threshold = 0.53
print("\nThreshold\n: ", threshold)
Y_pred = (Y_pred_proba > threshold).astype(np.int_)

# Count how many are predicted as 0 and 1
print("Predicted as 1: ", np.count_nonzero(Y_pred))
print("Predicted as 0: ", len(Y_pred) - np.count_nonzero(Y_pred))

# Compute the statistics
cmatrix = sklearn.metrics.confusion_matrix(test_labels, Y_pred)
print("Confusion Matrix:")
print(cmatrix)

accuracy = sklearn.metrics.accuracy_score(test_labels, Y_pred)
print("Accuracy: {:.3f}".format(accuracy))

precision = sklearn.metrics.precision_score(test_labels, Y_pred)
print("Precision: {:.3f}".format(precision))

recall = sklearn.metrics.recall_score(test_labels, Y_pred)
print("Recall: {:.3f}".format(recall))

F1 = sklearn.metrics.f1_score(test_labels, Y_pred)
print("F1: {:.3f}".format(recall))

# Compute a precision & recall graph
precisions, recalls, thresholds = \
    sklearn.metrics.precision_recall_curve(test_labels, Y_pred_proba)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.legend(loc="center left")
plt.xlabel("Threshold")
plt.show()

# Plot a ROC curve (Receiver Operating Characteristic)
# Compares true positive rate with false positive rate
fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, Y_pred_proba)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()
auc_score = sklearn.metrics.roc_auc_score(test_labels, Y_pred_proba)
print("AUC score: {:.3f}".format(auc_score))

# Predict new labels for training data
Y_pred_proba_training = model.predict(train_data[cols])
auc_score_training = sklearn.metrics.roc_auc_score(\
    train_labels, Y_pred_proba_training)
print("Training AUC score: {:.3f}".format(auc_score_training))

# Create a list of the abs(coeff) by feature
coeff_abs_list = []
for idx in range(len(model.coef_)):
    coeff_abs_list.append( (abs(model.coef_[idx]), cols[idx]) )

# Sort the list
coeff_abs_list.sort(reverse=True)

# Print the coefficients in order
for idx in range(len(model.coef_)):
    print("Feature: {:26s} abs(coef): {:.4f}".format(coeff_abs_list[idx][1], coeff_abs_list[idx][0]))
############################################################

# saving reesults to csv file
results = pd.read_csv("test.csv", header=0)

#inputing results to dataframe test_lables
results['Survived'] = Y_pred
cols = ['PassengerId', 'Survived']
results = results[cols]
results = results.set_index('PassengerId')
print(results.columns)
print(results)
results.to_csv('Results.csv')