import argparse
import numpy as np
from tslda import TSLDA
from tslda_data import tslda_data
from sklearn import svm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--K', type=int, default=10, help='# of topics.')
    parser.add_argument('-s', '--S', type=int, default=3, help='# of sentiments')
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help='Dirichlet prior vectors.')
    parser.add_argument('-b', '--beta', type=float, default=0.01, help='Dirichlet prior vectors.')
    parser.add_argument('-g', '--gamma', type=float, default=0.01, help='Dirichlet prior vectors.')
    parser.add_argument('-l', '--lam', type=float, default=0.1, help='Dirichlet prior vectors.')
    parser.add_argument('-p', '--test-proportion', type=float, default=0.2, help='Proportion of test set')
    return parser.parse_args()


def main():
    args = parse_args()
    messages, prices = tslda_data()
    documents = [doc for msg in messages for doc in msg]
    tslda = TSLDA(documents, args.alpha, args.beta, args.gamma, args.lam, args.K, args.S)

    # prepare data for classification training
    n = len(messages)
    n_train = n - int(n * args.test_proportion)
    clf = svm.SVC()
    X, y = [], []
    for t in range(2, n):
        price_feature = np.concatenate([np.eye(2)[prices[t-1]], np.eye(2)[prices[t-2]]])
        tslda_feature = np.concatenate([tslda(messages[t]), tslda(messages[t-1])])
        X.append(np.concatenate([price_feature, tslda_feature]))
        y.append(prices[t])

    clf.fit(X[:n_train], y[:n_train])
    print('accuracy:', np.mean(y[n_train:] == clf.predict(X[n_train:])))


if __name__ == '__main__':
    main()
