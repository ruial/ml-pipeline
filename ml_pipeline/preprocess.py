import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

EXTRA_FEATURES = ['count.url', 'count.hashtag', 'count.reply', 'count.exclamation',
                  'count.question', 'has.ellipsis', 'count.caps']

def add_features(df: pd.DataFrame):
  # remove accents
  df['text'] = df['text'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

  # some additional features
  df['count.url'] = df['text'].str.count('http')
  df['count.hashtag'] = df['text'].str.count('#')
  df['count.reply'] = df['text'].str.count('@')
  df['count.exclamation'] = df['text'].str.count('!')
  df['count.question'] = df['text'].str.count(r'\?')
  df['count.caps'] = df['text'].str.count(r'\b[A-Z]+\b')
  df['has.ellipsis'] = df['text'].str.contains('...', regex=False).astype(int)
  df['count.words'] = df['text'].str.count(r'[\w\-"\']+')
  df['average.word.length'] = df['text'].str.split().apply(lambda words: np.mean([len(word) for word in words])).astype(int)

  # add the keyword to the text
  df['text'] =  'keyword_' + df['keyword'].fillna('').str.replace('%20', '_') + ' ' + df['text']


def vectorize(df: pd.DataFrame, vectorizer: CountVectorizer):
  vector = vectorizer.transform(df['text'])
  return hstack((vector, df[EXTRA_FEATURES].values)).tocsr()
