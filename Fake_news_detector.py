import numpy as np 
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import itertools
import matplotlib.pyplot as plt

test  = pd.read_csv ("test.csv")
submit  = pd.read_csv ("submit.csv")

train = pd.read_csv("test.csv")

train.head()

print(f"Train Shape : {train.shape}")
print(f"Test Shape : {test.shape}")
print(f"Submit Shape : {submit.shape}")

train.info()

train.isnull().sum()

train.dtypes.value_counts()

test=test.fillna(' ')
train=train.fillna(' ')

test['total']=test['title']+' '+test['author']+' '+test['text']
train['total']=train['title']+' '+train['author']+' '+train['text']


train.info()
train.head()