from collections import defaultdict
import numpy as np


class TSLDA:
    def __init__(self, documents, alpha, beta, gamma, lam, K, S=3):
        self.documents = documents
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam = lam
        self.n_topics = K
        self.n_sentiments = S
        self.counter = defaultdict(lambda: 0)  # key: (sentence, topic, sentiment, word, category)
        self.group_by_topic = [dict() for _ in range(self.n_topics)]
        self.group_by_sentiment = [dict() for _ in range(self.n_sentiments)]
        self.pretrain()

    def pretrain(self):
        for _ in range(1000):
            self.gibbs_sampling()

    def gibbs_sampling(self):
        for doc in self.documents:
            for sent in doc.sentences:
                # sent.topic, sent.sentiment = multinomial()
                p = np.zeros((self.n_topics, self.n_sentiments))
                for a in range(self.n_topics):
                    for b in range(self.n_sentiments):
                        Za = len(self.group_by_topic[a]) - int(sent.topic == a)
                        Zb = len(self.group_by_sentiment[b]) - int(sent.sentiment == b)
                        # TODO
                        p[a, b] = (Za + self.beta) * (Zb + self.gamma) * () / () * () / ()
                if sent.topic is not None:
                    self.group_by_topic[sent.topic].pop(sent)
                if sent.sentiment is not None:
                    self.group_by_sentiment[sent.sentiment].pop(sent)
                p /= p.sum()
                ind = np.random.multinomial(1, p.flatten()).argmax()
                sent.topic = ind // self.n_sentiments
                sent.sentiment = ind % self.n_sentiments

    def __call__(self, documents):
        weights = np.zeros((self.n_topics, self.n_sentiments))
        for doc in documents:
            for sent in doc.sentences:
                weights[sent.topic, sent.sentiment] += 1
        weights /= weights.sum()
        return weights.flatten()
