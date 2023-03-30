import pandas as pd
import numpy as nd
from sklearn.naive_bayes import MultinomialNB 
#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
import csv
import os
import re
 


spam = pd.read_csv("C:\\Users\\bryso\\OneDrive\\Desktop\\Junior IS\\spam.csv")
 
spam = spam[["Label", "Message"]]


print(spam.shape)
spam.head()

spam['Label'].value_counts(normalize=True)
data_randomized = spam.sample(frac=1, random_state=1)
training_test_index = round(len(data_randomized) * 0.8)
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

print(training_set.shape)
print(test_set.shape)


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

print (len(vocabulary)) #number of unique words

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
#print (word_counts)
#print (counts)

training_set_clean = pd.concat([training_set, word_counts], axis=1)
training_set_clean.head()  
#print (training_set_clean)



spam_messages = training_set_clean[training_set_clean['Label'] == 'spam']
ham_messages = training_set_clean[training_set_clean['Label'] == 'ham']

p_spam = len(spam_messages) / len(training_set_clean)
p_ham = len(ham_messages) / len(training_set_clean)

n_words_per_spam_message = spam_messages['Message'].apply(len)
n_spam = n_words_per_spam_message.sum()

n_words_per_ham_message = ham_messages['Message'].apply(len)
n_ham = n_words_per_ham_message.sum()

n_vocabulary = len(vocabulary)

alpha = 1

# Initiate parameters
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}

# Calculate parameters
for word in vocabulary:
   n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
   p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
   parameters_spam[word] = p_word_given_spam

   n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
   p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
   parameters_ham[word] = p_word_given_ham

   

def classify(text):
   
   text = re.sub('\W', ' ', str(text))
   text = text.lower().split()

   p_spam_given_text = p_spam
   p_ham_given_text = p_ham

   for word in text:
      if word in parameters_spam:
         p_spam_given_text *= parameters_spam[word]

      if word in parameters_ham: 
         p_ham_given_text *= parameters_ham[word]

   print('P(Spam|text):', p_spam_given_text)
   print('P(Ham|text):', p_ham_given_text)

   if p_ham_given_text > p_spam_given_text:
      print('Label: Ham')
   elif p_ham_given_text < p_spam_given_text:
      print('Label: Spam')
   else:
      print('Equal proabilities, have a human classify this!')


def classify_test_set(text):
   

   text = re.sub('\W', ' ', str(text))
   text = text.lower().split()

   p_spam_given_text = p_spam
   p_ham_given_text = p_ham

   for word in text:
      if word in parameters_spam:
         p_spam_given_text *= parameters_spam[word]

      if word in parameters_ham:
         p_ham_given_text *= parameters_ham[word]

   if p_ham_given_text > p_spam_given_text:
      return 'ham'
   elif p_spam_given_text > p_ham_given_text:
      return 'spam'
   else:
      return 'needs human classification'
  

test_set['predicted'] = test_set['Message'].apply(classify_test_set)
test_set.head()

correct = 0
total = test_set.shape[0]

for row in test_set.iterrows():
   row = row[1]
   if row['Label'] == row['predicted']:
      correct += 1

print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)



#Potientially Useful 

#s = pd.Series(range(5))
#s.where(s > 0)
#df.query('A > B')
#pandas.DataFrame.lt
#DataFrame.transform
#casum = counts.sum()
#pd.DataFrame.sort_values(by=casum, ascending=False)
#pd.DataFrame.select_dtypes(exclude='bool')
#DataFrame.isin(values)
#hi_counts.loc['Total' , :]= hi_counts.sum(axis=0) 
#pd.DataFrame.groupby('team')['points'].apply(lambda grp: grp.nlargest(2).sum())

#vocab_words_spam = []

##for sentence in train_spam:
#   sentence_as_list = sentence.split()
#   for word in sentence_as_list:
#       vocab_words_spam.append(word)     
        
#print(vocab_words_spam)
#vocab_unique_words_spam = list(dict.fromkeys(vocab_words_spam))
#print(vocab_unique_words_spam)

#hiword_counts = pd.DataFrame(word_counts_per_message)
#hiword_counts.head()


#hiword_counts.loc['Total' , :]= hiword_counts.sum(axis=0) 
#hiword_counts[hiword_counts[word].str.len()>3]

#for word in hiword_counts:
#   if hiword_counts[hiword_counts.sum < 5]:
#      hiword_counts.pop(word)

#print (hiword_counts)

#hiword_counts = hiword_counts.astype(int)