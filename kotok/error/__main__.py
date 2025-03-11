import os
import logging
import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='command')

    model_default = 'klue/bert-base'
    cache_default = os.path.join('cache')
    data_default = os.path.join('data', 'labeled_error.json')

    data = subparsers.add_parser('data')
    data.add_argument('-m', '--model', type=str, default=model_default)
    data.add_argument('-c', '--cache', type=str, default=cache_default)
    data.add_argument('-i', '--input', type=str, default=os.path.join('data', 'txt_error'))
    data.add_argument('-n', '--normalize_mode', type=str, default=None)
    data.add_argument('-o', '--output', type=str, default=data_default)
    data.add_argument('-s', '--split', type=float, default=0.8)

    train = subparsers.add_parser('train')
    train.add_argument('-m', '--model', type=str, default=model_default)
    train.add_argument('-c', '--cache', type=str, default=cache_default)
    train.add_argument('-d', '--data', type=str, default=data_default)
    train.add_argument('-o','--output', type=str, default='kotok_error_model')
    train.add_argument('-l', '--logs', type=str, default='logs')

    inference = subparsers.add_parser('inference')
    inference.add_argument('-cm', '--classification_model', type=str, default='kotok_error_model')
    inference.add_argument('-m', '--model', type=str, default=model_default)
    inference.add_argument('-c', '--cache', type=str, default=cache_default)

    return parser

def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    args = make_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == 'data':
        from .data import data
        data(**args.__dict__)
    elif args.command == 'train':
        from .train import train
        train(args)
    elif args.command == 'inference':
        from .inference import inference
        inference(**args.__dict__)

if __name__ == '__main__':
    main()
