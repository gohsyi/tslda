import os
import numpy as np
from pandas import read_csv
from datetime import datetime

import stanza

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# stanza.download('en')


class TSLDAData:
    def __init__(self):
        self._historical = read_csv('data/historical.csv')
        self._message = read_csv('data/message.csv', header=None)
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,lemma')  # lemmatization
        self.message = {}
        self.historical = {}
        self.preprocess()

    def preprocess(self):
        for i, (date, msg) in self._message.iterrows():
            date = datetime.strptime(date, '%Y-%m-%d')
            msg = msg.strip('.')  # remove stop words
            if date not in self.message:
                self.message[date] = []
            self.message[date].append(self.nlp(msg))
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

    def __call__(self):
        return self.historical, self.message


tslda_data = TSLDAData()
