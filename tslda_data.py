import os
import numpy as np
from pandas import read_csv
from datetime import datetime
from collections import OrderedDict
from nltk.corpus import wordnet

import stanza

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# stanza.download('en')



class Sentence:
    def __init__(self, words):
        self.words = words
        self.topic = None
        self.sentiment = None


class Word:
    def __init__(self, lemmatized):
        self.word = lemmatized
        self.part = wordnet.synsets(w)[0].pos()
        self.category = None


class TSLDAData:
    def __init__(self):
        self._historical = read_csv('data/historical.csv')
        self._message = read_csv('data/message.csv', header=None)
        self.opinion_words = read_csv('data/SentiWordNet_3.0.0.txt', comment='#', sep='\t', header=None)
        self.lemmatization = stanza.Pipeline(lang='en', processors='tokenize,mwt,lemma,pos')  # lemmatization
        self.message = OrderedDict()
        self.historical = OrderedDict()
        self.preprocess()

    def preprocess(self):
        self._historical = self._historical.sort_index(ascending=False)  # in the order of time
        last_day = None
        for i, historical in self._historical.iterrows():
            if not last_day:
                last_day = historical['Adj Close']
            else:
                date = datetime.strptime(historical['Date'], '%Y-%m-%d')
                adj_close = historical['Adj Close']  # the adjusted close prices
                self.historical[date] = int(adj_close < last_day)
                last_day = adj_close
        for i, (date, msg) in self._message.iterrows():
            date = datetime.strptime(date, '%Y-%m-%d')
            if date in self.historical:  # only store messages on transaction days
                msg = msg.strip('.')  # remove stop words
                if date not in self.message:
                    self.message[date] = []
                self.message[date].append(self.lemmatization(msg))

    def __call__(self):
        return list(self.historical.values()), list(self.message.values())


tslda_data = TSLDAData()
