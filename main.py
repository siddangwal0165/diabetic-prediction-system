import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#getting the file
#pandas convert the data into a structured dataset
data_set=pd.read_csv('D:\diabetes\diabetes_dataset.csv')
#head function -by this get the top 5 values of dataset
print(data_set.head())
#SHAPE FUNCTION GIVES NO OF ROWS AND COLOUMS IN DATASET
print(data_set.shape)

#describe function--by this we get the statistical values like mean ,median etc of dataset
print('MEAN MEDIAN OF THE WHOLE DATASET')
print(data_set.describe())

print('no of daibetic and non daibetic patients')
# 0--non daibetic
# 1--daibetic

print(data_set['Outcome'].value_counts())

print('mean median of the basis of daibetic and non daibetic ')
print(data_set.groupby('Outcome').mean())

# separting the feature and labels of dataset
featureS=data_set.drop(columns = 'Outcome', axis=1)#removing the outcome coloum from dataset and store in x;
#axis=1 for droping coloum and axis=0 use for dropping the row
label=data_set['Outcome']#storing label in y


#22
scaler=StandardScaler()
scaler.fit(featureS)
standard_data=scaler.transform(featureS)
#print('------------------standard data---------------------------')
#print(standard_data)

print('data spliting')

#x_train the feateure(input) for training ---
#y_train the label(target) for training
#x_test  input feature for testing dataset
#y_test  label(target) for testing
x_train,x_test,y_train,y_test=train_test_split(featureS,label,test_size=0.2,stratify=label,random_state=2)

#train test split give 4 outputs and this store in the above variable
#test size=0.2 it means 20% of data is for testing and rest 80% for training
#stratify is use to equate the labels if we not this there may be a case that all the positive 
#result goes to one and all the negative goes to another one
#random state its type we can use any no like 1 ,3 etc
print('shappeeeeeeeeeeeeeeee')
print(x_train.shape)
classifier=svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)


x_test_prediction=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('accuracy is',test_data_accuracy)

