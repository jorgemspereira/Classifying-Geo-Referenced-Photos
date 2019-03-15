import argparse
import os
import random as rn
import warnings

import numpy as np
import tensorflow as tf
from keras.backend import set_session

from helpers.arguments import Mode, Dataset, Method
from helpers.dataset import get_train_dataset_info
from helpers.training import train_test_model_cv, train_test_model_split

RANDOM_SEED = 20
BATCH_SIZE = 8
N_FOLDS = 10
EPOCHS = 50


def initial_configs():
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    rn.seed(RANDOM_SEED)
    np.set_printoptions(threshold=np.inf)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # warnings.filterwarnings("ignore", category=UserWarning)
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
    return vars(parser.parse_args())


def train_test_model(args):
    info = get_train_dataset_info(args['dataset'])
    is_binary = args['dataset'] != Dataset.flood_severity

    if args['method'] == Method.cross_validation:
        train_test_model_cv(info, args, is_binary, RANDOM_SEED, BATCH_SIZE, EPOCHS, N_FOLDS)
    if args['method'] == Method.train_test_split:
        train_test_model_split(info, args, is_binary, RANDOM_SEED, BATCH_SIZE, EPOCHS)


def main():
    initial_configs()
    parsed_args = parse_args()
    train_test_model(parsed_args)

    # draw_class_activation_map(parsed_args)


if __name__ == "__main__":
    main()
