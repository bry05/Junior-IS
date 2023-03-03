import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import csv
import os


spam = pd.read_csv("C:\\Users\\bryso\\OneDrive\\Desktop\\Junior IS\\60Tests.csv")

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

count_vec = CountVectorizer()

               
word_count_vec = count_vec.fit_transform(spam.Message)
word_count_vec.shape

print(count_vec.get_feature_names_out())    

counts = pd.DataFrame(word_count_vec.toarray(), index=['1', '2', '3', '4', '5', '6', '7','8','9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60'],columns=count_vec.get_feature_names_out())
counts.loc['Total',:]= counts.sum(axis=0)

print (counts)
