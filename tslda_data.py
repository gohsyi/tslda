import re
import json
from pandas import read_csv
from datetime import datetime
from collections import defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

opinion_words = set(read_csv(
    'data/SentiWordNet_3.0.0.txt', comment='#', sep='\t', header=None)[4].tolist())


def date_is_selected(date):
    return date.year == 2020 and date.month > 6


class Document:
    def __init__(self, raw):
        self.sentences = [Sentence(sent.strip('.')) for sent in nltk.sent_tokenize(raw)]


class Sentence:
    def __init__(self, sentence):
        # remove punctuations
        sentence = re.sub(r'[^\w\s.]', '', sentence)
        # tokenize
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
        self.lemma = lemmatizer.lemmatize(word.lower())
        synsets = wordnet.synsets(self.lemma)
        self.part = synsets[0].pos() if len(synsets) > 0 else None
        self.category = None


class TSLDAData:
    def __init__(self, stock):
        self._prices = read_csv(f'data/prices/{stock}_prices.csv')
        self._messages = json.load(open(f'data/news/{stock}_news.json'))
        self.opinion_words = read_csv('data/SentiWordNet_3.0.0.txt', comment='#', sep='\t', header=None)
        self.all_messages = defaultdict(lambda: list())
        self.messages = list()
        self.prices = list()
        self.dates = list()
        self.preprocess()

    def preprocess(self):
        for idx in self._messages['date'].keys():
            date = datetime.strptime(self._messages['date'][idx], '%Y/%m/%d')
            if date_is_selected(date):
                self.all_messages[date].append(Document(self._messages['text'][idx]))

        self._prices = self._prices.sort_index(ascending=False)  # in the order of time
        last_day = None
        for i, p in self._prices.iterrows():
            if not last_day:
                last_day = p['price']
            else:
                date = datetime.fromisoformat(p['date'])
                adj_close = p['price']  # the adjusted close prices
                if date in self.all_messages.keys():
                    self.prices.append(int(adj_close < last_day))
                    self.messages.append(self.all_messages[date])
                    self.dates.append(date)
                last_day = adj_close

    def __call__(self):
        return self.messages, self.prices, self.dates
