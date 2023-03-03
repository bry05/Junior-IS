import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

spam = pd.read_csv("C:\\Users\\bryso\\OneDrive\\Desktop\\Junior IS\\spam.csv")

spam = spam[["Label", "Message"]]
x = spam['Message']
y = spam["Label"]
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2)

cv = CountVectorizer()
features = cv.fit_transform(x_train)

model = svm.SVC()
model.fit(features,y_train)

features_test = cv.transform(x_test)
print("Accuracy: {}".format(model.score(features_test,y_test)))

#print 

