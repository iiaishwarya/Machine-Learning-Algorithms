import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from functools import reduce
import numpy as np
import operator

"""## **Load the Data**"""

def loadData():
    X_train = tokenized(pd.read_csv("traindata.txt", header=None)[0])
    y_train = pd.read_csv("trainlabels.txt", header=None)
    X_test = tokenized(pd.read_csv("testdata.txt", header=None)[0])
    y_test = pd.read_csv("testlabels.txt", header=None)
    stop_words = tokenized(pd.read_csv("stoplist.txt", header=None)[0])
    return X_train, y_train, X_test, y_test, stop_words

"""## **Tokenize the words**"""

def tokenized(text):
  tokenized_text = text.apply(word_tokenize)
  return list(tokenized_text)

"""## **Removing Stopwords**"""

def removeStopwords(data, stop_words):
    stop_words = reduce(operator.add, stop_words)
    final = []
    for row in data:
      for word in row: 
        if word not in stop_words and word not in final:
            final.append(word)
    final.sort()
    return final

"""## **Features Matrix**"""

def term_matrix(data, dictionary, y):
  feature_matrix = []
  for row in data:
      vector = dict.fromkeys(dictionary, 0)
      for word in row:
          if word in vector:
              vector[word] = 1
      feature_matrix.append(vector)
  feature_matrix = pd.DataFrame(feature_matrix)
  feature_matrix['target'] = y
  return feature_matrix

def getCount(df, column, val):
    total_wise = df[(df[column] == val) & (df['target'] == 0)].shape[0]
    total_future = df[(df[column] == val) & (df['target'] == 1)].shape[0]
    return total_wise, total_future, total_wise + total_future

def laplace_smoothing(count, total):
  pr = (count + 1)/(total + 2)
  return 0 if pr == 0 else pr

def probability(count, total):
  pr = count / total
  return 0 if pr == 0 else pr

"""## **Calculate Probability**"""

def wordsProbability(features, vocabulary):
    probability_words = {}
    for word in vocabulary:
        dictionary = { word: { 1: {'wise': {}, 'future':{} }, 0: {'wise':{},'future':{}}}}
        # 1: present, 0: absent
        for value in range(2):
            total_wise, total_future , total = getCount(features, word, value)
            p_w = laplace_smoothing(total_wise, total)
            dictionary[word][value]['wise'] = p_w
            dictionary[word][value]['future'] = 1 - p_w
        probability_words.update(dictionary)

    total_wise = features[features['target'] == 0].shape[0]
    total_future = features[features['target'] == 1].shape[0]
    probability_wise = probability(total_wise, total_wise + total_future)
    probability_future = 1 - probability_wise

    return probability_words, probability_wise, probability_future

X_train, y_train, X_test, y_test, stop_words = loadData()

vocabulary = removeStopwords(X_train, stop_words)
featureTrain = term_matrix(X_train, vocabulary, y_train)
featureTest = term_matrix(X_test, vocabulary, y_test)
probability_dictionary, p_w, p_f = wordsProbability(featureTrain, vocabulary)

class NaiveBayes:
  def __init__(self, prob_dictionary, p_w, p_f):
    self.prob_dictionary = prob_dictionary
    self.p_w = p_w
    self.p_f = p_f

  def predict(self, data):
    labels = data.iloc[:, -1]
    data = data.drop('target', 1)
    correct = 0
    for i in range(len(data)):
      wise_prob, future_prob = 1, 1
      for word in self.prob_dictionary.keys():
          if data.iloc[i][word] == 1:
              wise_prob *= self.prob_dictionary[word][1]['wise']
              future_prob *= self.prob_dictionary[word][1]['future']

      wise_prob *= self.p_w
      future_prob *= self.p_f
      label = 0 if wise_prob > future_prob else 1
      if(label == labels.iloc[i]):
          correct += 1
    accuracy = probability(correct, len(data))
    return accuracy

nb = NaiveBayes(probability_dictionary, p_w, p_f)

accuracy_train = nb.predict(featureTrain)
accuracy_test = nb.predict(featureTest)

print("Training Accurary", accuracy_train)
print("Testing Accuracy", accuracy_test)
