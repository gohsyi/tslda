from collections import defaultdict
from tqdm import trange
import numpy as np


class TSLDA:
    def __init__(self, documents, alpha, beta, gamma, lam, T, K, S=3):
        self.documents = documents
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam = lam
        self.n_topics = K
        self.n_sentiments = S
        self.counter = [
            [[defaultdict(lambda: 0) for _ in range(3)]
             for _ in range(self.n_sentiments)]
            for _ in range(self.n_topics)]  # shape is (K x S x 3), for W^{a,b}_{*,*,v,c}
        self.group_by_topic = np.zeros(self.n_topics)
        self.group_by_sentiment = np.zeros(self.n_sentiments)
        self.n_words = self.calc_num_words()
        self.pretrain(T)

    def calc_num_words(self):
        wordset = set()
        for doc in self.documents:
            for sent in doc.sentences:
                wordset.update(sent.wordset)
        return len(wordset)

    def pretrain(self, T):
        for _ in trange(T):
            self.gibbs_sampling()

    def gibbs_sampling(self):
        for doc in self.documents:
            for sent in doc.sentences:
                # sent.topic, sent.sentiment = multinomial()
                p = np.zeros((self.n_topics, self.n_sentiments))
                for a in range(self.n_topics):
                    for b in range(self.n_sentiments):
                        Za = self.group_by_topic[a] - int(sent.topic == a)
                        Zb = self.group_by_sentiment[b] - int(sent.sentiment == b)
                        num1 = np.prod([
                            np.prod([
                                sum(self.counter[a][b_][1][key] for b_ in range(self.n_sentiments))
                                - int(sent.topic == a) * sent.wordcalc[1][key]
                                + self.alpha + j
                                for j in range(sent.wordcalc[1][key])
                            ]) for key in sent.wordset])
                        den1 = np.prod([
                            sum(sum(self.counter[a][b_][1].values()) for b_ in range(self.n_sentiments))
                            - int(sent.topic == a) * sum(sent.wordcalc[1].values())
                            + self.n_words * (self.alpha + j)
                            for j in range(sum(sent.wordcalc[1].values()))])
                        num2 = np.prod([
                            np.prod([
                                self.counter[a][b][2][key]
                                - int(sent.topic == a and sent.sentiment == b) * sent.wordcalc[2][key]
                                + self.lam + j
                                for j in range(sent.wordcalc[2][key])
                            ]) for key in sent.wordset])
                        den2 = np.prod([
                            sum(self.counter[a][b][2].values())
                            + self.n_words * (self.lam + j)
                            for j in range(sum(sent.wordcalc[2].values()))])
                        p[a, b] = (Za + self.beta) * (Zb + self.gamma) * num1 / den1 * num2 / den2
                if sent.topic is not None:
                    self.group_by_topic[sent.topic] -= 1
                    self.group_by_sentiment[sent.sentiment] -= 1
                    for word in sent.words:
                        self.counter[sent.topic][sent.sentiment][word.category][word.lemma] -= 1
                p += 1e-7
                p /= p.sum()
                ind = np.random.multinomial(1, p.flatten()).argmax()
                sent.topic = ind // self.n_sentiments
                sent.sentiment = ind % self.n_sentiments
                self.group_by_topic[sent.topic] += 1
                self.group_by_sentiment[sent.sentiment] += 1
                for word in sent.words:
                    self.counter[sent.topic][sent.sentiment][word.category][word.lemma] += 1

    def __call__(self, documents):
        weights = np.zeros((self.n_topics, self.n_sentiments))
        for doc in documents:
            for sent in doc.sentences:
                weights[sent.topic, sent.sentiment] += 1
        weights /= weights.sum()
        return weights.flatten()
