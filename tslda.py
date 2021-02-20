


class TSLDA:
    def __init__(self, messages, alpha, beta, gamma, lam, K, S=3):
        self.messages = messages
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam = lam
        self.counter = defaultdict(lambda: 0)
        self.pretrain()

    def pretrain(self):
        for _ in range(1000):
            self.gibbs_sampling()

    def gibbs_sampling(self):
        pass

    def __call__(self):
        return self.weights.flatten()
