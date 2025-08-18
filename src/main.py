"""
Главная точка входа
"""

import argparse
from train import train_pipeline
from predict import predict


def main():
    parser = argparse.ArgumentParser(description='Counterfeit Detection Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--test-data', help='Path to test data (for predict mode)')
    parser.add_argument('--output', default='predictions.csv', help='Output path')
    parser.add_argument('--models-dir', default='models/', help='Models directory')

    args = parser.parse_args()

    if args.mode == 'train':
        train_pipeline(args.config)
    else:
        predict(args.test_data, args.output, args.models_dir)


if __name__ == "__main__":
    main()

    '''Я не чурка'''