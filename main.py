import argparse
import os
import random as rn

import numpy as np
import tensorflow as tf
import warnings
from keras.backend import set_session

from helpers.arguments import Mode, Dataset, Method, Model
from helpers.training import train_test_model_split, train_test_attention_guided_cnn, train_test_model_cv

RANDOM_SEED = 20
BATCH_SIZE = 8
N_FOLDS = 10
EPOCHS = 50


def initial_configs():
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    rn.seed(RANDOM_SEED)
    warnings.filterwarnings("ignore", category=UserWarning, module='keras')
    np.set_printoptions(threshold=np.inf)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)


# noinspection PyTypeChecker
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest='mode', choices=list(Mode), type=Mode.from_string, required=True)
    parser.add_argument("--dataset", dest='dataset', choices=list(Dataset), type=Dataset.from_string, required=True)
    parser.add_argument("--method", dest='method', choices=list(Method), type=Method.from_string, required=True)
    parser.add_argument("--model", dest='model', choices=list(Model), type=Model.from_string, required=True)
    parser.add_argument("--data-augmentation", dest='data_augmentation', action='store_true')
    parser.add_argument("--class-activation-map", dest='class_activation_map', action='store_true')
    parser.add_argument("--print-classifications", dest='print_classifications', action='store_true')
    parser.add_argument("--mix-up", dest='mix_up', action='store_true')
    return vars(parser.parse_args())


def train_test_model(args):
    is_binary = args['dataset'] not in [Dataset.flood_severity_3_classes,
                                        Dataset.flood_severity_4_classes,
                                        Dataset.flood_severity_european_floods]

    args['random_seed'] = RANDOM_SEED
    args['batch_size'] = BATCH_SIZE
    args['is_binary'] = is_binary
    args['epochs'] = EPOCHS

    if args['method'] == Method.cross_validation:
        args['nr_folds'] = N_FOLDS

        if args['model'] == Model.attention_guided:
            train_test_attention_guided_cnn(args)
        elif args['model'] == Model.dense_net:
            train_test_model_cv(args)

    if args['method'] == Method.train_test_split:
        train_test_model_split(args)


def main():
    initial_configs()
    parsed_args = parse_args()
    train_test_model(parsed_args)


if __name__ == "__main__":
    main()
