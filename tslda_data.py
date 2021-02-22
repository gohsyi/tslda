import numpy as np
from pandas import read_csv
from datetime import datetime
from collections import OrderedDict, defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

opinion_words = set(read_csv(
    'data/SentiWordNet_3.0.0.txt', comment='#', sep='\t', header=None)[4].tolist())


class Document:
    def __init__(self, raw):
        self.sentences = [Sentence(sent.strip('.')) for sent in nltk.sent_tokenize(raw)]


class Sentence:
    def __init__(self, sentence):
        self.words = [Word(word) for word in nltk.word_tokenize(sentence)]
        self.wordset = set()
        self.wordcalc = [defaultdict(lambda: 0) for _ in range(3)]
        self.topic = None
        self.sentiment = None
        # categorization
        for i, word in enumerate(self.words):
            if word.part == 'n' and i > 0 and self.words[i - 1].part == 'n':
                # consecutive nouns
                word.category = 1
                self.words[i - 1].category = 1
            elif word.lemma in opinion_words:
                word.category = 2
            else:
                word.category = 0
        # construct word sets
        for word in self.words:
            self.wordset.add(word.lemma)  # for V_{d,m}
            self.wordcalc[word.category][word.lemma] += 1  # for W^{*,*}_{d,m,v,c}


class Word:
    def __init__(self, word):
        self.lemma = lemmatizer.lemmatize(word)
        synsets = wordnet.synsets(self.lemma)
        self.part = synsets[0].pos() if len(synsets) > 0 else None
        self.category = None


class TSLDAData:
    def __init__(self):
        self._prices = read_csv('data/historical.csv')
        self._messages = read_csv('data/message.csv', header=None)
        self.opinion_words = read_csv('data/SentiWordNet_3.0.0.txt', comment='#', sep='\t', header=None)
        self.all_messages = defaultdict(lambda: list())
        self.messages = list()
        self.prices = list()
        self.preprocess()

    def preprocess(self):
        for i, (date, msg) in self._messages.iterrows():
            date = datetime.strptime(date, '%Y-%m-%d')
            self.all_messages[date].append(Document(msg))

        self._prices = self._prices.sort_index(ascending=False)  # in the order of time
        last_day = None
        for i, p in self._prices.iterrows():
            if not last_day:
                last_day = p['Adj Close']
            else:
                date = datetime.strptime(p['Date'], '%Y-%m-%d')
                adj_close = p['Adj Close']  # the adjusted close prices
                self.prices.append(int(adj_close < last_day))
                self.messages.append(self.all_messages[date])
                last_day = adj_close

    def __call__(self):
        return self.messages, self.prices


tslda_data = TSLDAData()
