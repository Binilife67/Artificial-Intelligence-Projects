import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import streamlit as st

dataf = pd.read_csv('Files/spam.csv')
dataf.drop_duplicates(inplace=True)
# print(dataf.isnull().sum())
dataf['Category'] = dataf['Category'].replace(['ham', 'spam'], ['Not spam', 'Spam'])

dataf_mess = dataf['Message']
dataf_cat = dataf['Category']

train_message, test_message, train_category, test_category = train_test_split(dataf_mess, dataf_cat, test_size=0.4)

train = train_message.to_frame().join(train_category) # pd.conc([train_message, train_category], axis=1) - for series
test = test_message.to_frame().join(test_category)

cv = CountVectorizer(stop_words='english')
train_mess_features = cv.fit_transform(train_message)
test_mess_features = cv.transform(test_message)

mnv = MultinomialNB()
mnv.fit(train_mess_features, train_category)

def detect_spam_message(message):
  if isinstance(message, str):
    message = [message]

  message = cv.transform(message)
  result = mnv.predict(message)

  return str(result[0])

st.header('Spam detection')
message = st.text_input('Enter message here')
if st.button('Check'):
  result = detect_spam_message(message)
  output = result[0] if len(result) == 1 else result
  st.success(output) # or st.write but not st.text