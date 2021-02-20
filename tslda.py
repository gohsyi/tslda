from collections import defaultdict
import numpy as np  # for np.random.dirichlet(alpha, size)


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
        self.pretrain()

    def pretrain(self):
        for _ in range(1000):
            self.gibbs_sampling()

    def gibbs_sampling(self):
        for doc in self.documents:
            for sent in doc.sentences:
                # sent.topic, sent.sentiment = multinomial()
                pass

    def __call__(self, documents):
        weights = np.zeros((self.n_topics, self.n_sentiments))
        for doc in documents:
            for sent in doc.sentences:
                weights[sent.topic, sent.sentiment] += 1
        weights /= weights.sum()
        return weights.flatten()
