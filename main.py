import argparse
from tslda import TSLDA
from tslda_data import tslda_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--K', type=int, default=10, help='# of topics.')
    parser.add_argument('-s', '--S', type=int, default=3, help='# of sentiments')
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help='Dirichlet prior vectors.')
    parser.add_argument('-b', '--beta', type=float, default=0.01, help='Dirichlet prior vectors.')
    parser.add_argument('-g', '--gamma', type=float, default=0.01, help='Dirichlet prior vectors.')
    parser.add_argument('-l', '--lambda', type=float, default=0.1, help='Dirichlet prior vectors.')
    return parser.parse_args()


def main():
    args = parse_args()
    historical, messages = tslda_data()
    tslda = TSLDA(messages, args.alpha, args.beta, args.gamma, args.lam, args.K, args.S)

    # prepare data for classification training
    pass



if __name__ == '__main__':
    main()
