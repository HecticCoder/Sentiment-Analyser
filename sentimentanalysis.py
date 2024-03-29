import pandas as pd
import streamlit as st

st.title('SENTIMENT ANALYSER')

df = pd.read_csv('https://raw.githubusercontent.com/HecticCoder/Sentiment-Analyser/main/Test.csv')

x = df.iloc[:,1].values
y = df.iloc[:,0].values

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
x = df['text'].astype('U').values
y = df['label']
x = tfidf.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0, stratify = y)

from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(x_train, y_train)

select = st.text_input('Enter your review')
select.replace('\d+', '')
select.replace('[^\w\s]', '')
vect = tfidf.transform([select])
op = clf.predict(vect)

if op == 1:
  op = 'Positive'

else:
  op = 'Negative'

if st.button('Predict'):
  st.title(op)
