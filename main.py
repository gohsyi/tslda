import argparse
import numpy as np
import pandas as pd
from tslda import TSLDA
from tslda_data import TSLDAData
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
    parser.add_argument('-t', '--T', type=int, default=100, help='Iterations for Gibbs sampling')
    parser.add_argument('--stock', type=str, default='ebay', help='Stock name.')
    return parser.parse_args()


def main():
    args = parse_args()
    messages, prices, dates = TSLDAData(args.stock.upper())()
    documents = [doc for msg in messages for doc in msg]
    tslda = TSLDA(documents, args.alpha, args.beta, args.gamma, args.lam, args.T, args.K, args.S)

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

    pd.DataFrame([[date] + tslda(news).tolist() for news, date in zip(messages, dates)]).to_csv(
        f'data/news/{args.stock.upper()}_news.csv', index=False)

    clf.fit(X[:n_train], y[:n_train])
    print('accuracy:', np.mean(y[n_train:] == clf.predict(X[n_train:])))


if __name__ == '__main__':
    main()
