import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('CLEF.csv', delimiter = '\t')

for col in ['public_id','title','text', 'our rating']:
    dataset[col] = dataset[col].astype('category')


dataset = pd.get_dummies(data=dataset,columns=['public_id','title','text'])


from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
dataset['our rating'] = labelencoder.fit_transform(dataset['our rating'])

X=dataset.drop(columns=['our rating'])
y=dataset['our rating']

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)


model=DecisionTreeClassifier()

model.fit(X_train,y_train)

predictions=model.predict(X_test)

score=accuracy_score(y_test, predictions)
print(score)
