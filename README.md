# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess data: Read the dataset, handle encoding, check for nulls, and split into features (messages) and labels (spam/ham).
2. Split dataset: Divide the data into training and testing sets using train_test_split.
3. Vectorize text: Convert text data into numerical format using CountVectorizer.
4. Train and predict: Fit a Support Vector Classifier (SVC) on the training data and predict on the test set.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SImon Malachi S
RegisterNumber: 212224040318  
*/
```

```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred


```
## Output:


![image](https://github.com/user-attachments/assets/538d8fbe-8e7b-4e25-a1c5-c9c29f5e7051)


![image](https://github.com/user-attachments/assets/09567662-fd71-4450-9768-f873fb08fb6f)


![image](https://github.com/user-attachments/assets/2cd45135-cb51-4ef4-871f-15dbea4e3cdc)



![image](https://github.com/user-attachments/assets/037adab5-04d0-41b9-b5a6-475ebac68ab6)





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
