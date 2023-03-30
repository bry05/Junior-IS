import pandas as pd
import numpy as nd
from sklearn.naive_bayes import MultinomialNB 
#from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import csv
import os
import re


spam = pd.read_csv("C:\\Users\\bryso\\OneDrive\\Desktop\\Junior IS\\16HAM.csv")

spam = spam[["Label", "Message"]]
spam['Label'].value_counts(normalize=True)
data_randomized = spam.sample(frac=1, random_state=1)
training_test_index = round(len(data_randomized) * 0.8)
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

training_set['Label'].value_counts(normalize=True)
test_set['Label'].value_counts(normalize=True)

training_set.head(3)
# After cleaning
training_set['Message'] = training_set['Message'].astype(str).str.replace( '\W', ' ', regex=True) 
training_set['Message'] = training_set['Message'].astype(str).str.lower()
training_set.head(3)


training_set['Message'] = training_set['Message'].astype(str).str.split()

vocabulary = []
for message in training_set['Message']:
   for word in message:
        vocabulary.append(word)

vocabulary = list(set(vocabulary))
 
len(vocabulary)

word_counts_per_message = {unique_word: [0] * len(training_set['Message']) for unique_word in vocabulary}

for index, message in enumerate(training_set['Message']):
   for word in message:
      word_counts_per_message[word][index] += 1



word_counts = pd.DataFrame(word_counts_per_message)
word_counts.head()
word_counts.loc['Total' , :]= word_counts.sum(axis=0) 


counts = pd.DataFrame.sum((word_counts))
#counts.head()
top = pd.Series(counts, word_counts_per_message).nlargest(50)
top = top.sort_values(ascending=False)


print (top)
