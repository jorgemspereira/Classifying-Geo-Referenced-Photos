import argparse
import math
import os
import random as rn
import warnings
from operator import itemgetter

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import metrics
from keras.applications.densenet import DenseNet201
from keras.backend import set_session
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.engine.saving import load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from helpers.arguments import Mode, Dataset, Method
from helpers.dataset import get_train_dataset_info, get_test_dataset_info

RANDOM_SEED = 20

BATCH_SIZE = 5
N_FOLDS = 10
EPOCHS = 30


def initial_configs():
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    rn.seed(RANDOM_SEED)
    np.set_printoptions(threshold=np.inf)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore", category=UserWarning)
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)


def create_model():
    base_model = DenseNet201(include_top=False, weights='imagenet')
    optimizer = Adam(lr=1e-4, amsgrad=True)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[metrics.binary_accuracy])

    return model


def get_callbacks(filepath):
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='auto', save_best_only=True)
    # cyclic_lr = CyclicLR(base_lr=1e-5, step_size=1500., max_lr=1e-4, mode='triangular2')

    def step_decay(epoch):
        initial_learning_rate, drop, epochs_drop = 1e-4, 0.5, 3.0
        return initial_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)
    scheduler_lr = LearningRateScheduler(schedule=step_decay, verbose=1)

    # return [early_stopping, checkpoint, cyclic_lr]

    return [early_stopping, scheduler_lr]


def get_training_and_validation_flow(df, directory, split_size=0.10):
    x, y = df.iloc[:, 0].values, df.iloc[:, 1].values
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=split_size, stratify=y, random_state=RANDOM_SEED)

    train_data_frame = pd.DataFrame(data={'filename': x_train, 'class': y_train})
    validation_data_frame = pd.DataFrame(data={'filename': x_val, 'class': y_val})

    train_generator = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True,
                                         zoom_range=(0.95, 1), brightness_range=(0.9, 1.1))
    trn_flow = train_generator.flow_from_dataframe(dataframe=train_data_frame, directory=directory,
                                                   target_size=(224, 224), class_mode="binary", shuffle=True,
                                                   has_ext=False, seed=RANDOM_SEED, batch_size=BATCH_SIZE)
    trn_flow.reset()

    val_generator = ImageDataGenerator(rescale=1. / 255)
    val_flow = val_generator.flow_from_dataframe(dataframe=validation_data_frame, directory=directory,
                                                 target_size=(224, 224), class_mode="binary", shuffle=True,
                                                 has_ext=False, seed=RANDOM_SEED, batch_size=1)
    val_flow.reset()

    return trn_flow, val_flow


def get_testing_flow(df, directory):
    image_generator = ImageDataGenerator(rescale=1. / 255)
    tst_flow = image_generator.flow_from_dataframe(dataframe=df, directory=directory, target_size=(224, 224),
                                                   class_mode="binary", shuffle=False, has_ext=False,
                                                   seed=RANDOM_SEED, batch_size=1)

    tst_flow.reset()

    return tst_flow


def train_test_model_split(train_directory, train_df, args):
    filepath = "weights/{}_split_2/weights.hdf5".format(args['dataset'])
    test_directory, test_df = get_test_dataset_info(args['dataset'])

    trn_flow, val_flow = get_training_and_validation_flow(train_df, train_directory, split_size=0.20)
    tst_flow = get_testing_flow(test_df, test_directory)

    if args['mode'] == Mode.train:
        model = create_model()
        model.fit_generator(generator=trn_flow, steps_per_epoch=(trn_flow.n // BATCH_SIZE),
                            validation_data=val_flow, validation_steps=val_flow.n,
                            callbacks=get_callbacks(filepath), epochs=EPOCHS)
    else:
        model = load_model(filepath)

    y_pred_prob = model.predict_generator(generator=tst_flow, verbose=1, steps=tst_flow.n)
    y_pred = np.where(y_pred_prob > 0.5, 1, 0).flatten()
    y_test = tst_flow.classes

    precision, recall, f_score, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    accuracy = accuracy_score(y_test, y_pred)

    # print_classifications(tst_flow, y_pred)

    print("F-Score ------------> {}".format(f_score))
    print("Precision ----------> {}".format(precision))
    print("Recall -------------> {}".format(recall))
    print("Accuracy -----------> {}".format(accuracy))

    calculate_average_precision_ks(list(zip(y_pred_prob.flatten().tolist(), y_test)))


def print_classifications(tst_flow, y_pred):
    for idx, el in enumerate(y_pred):
        green, red, end = '\033[92m', '\033[91m', '\033[0m'
        color = green if tst_flow.classes[idx] == el else red
        print(color + "{:<15} -> True: {} | Pred: {}".format(tst_flow.filenames[idx], tst_flow.classes[idx], el) + end)


def train_test_model_cv(directory, info, args):
    metrics_dict, y_pred_probs = {"precision": 0, "recall": 0, "f-score": 0, "accuracy": 0}, []
    x, y, fold_nr = info.iloc[:, 0].values, info.iloc[:, 1].values, 1
    k_fold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for train, test in k_fold.split(x, y):
        filepath = "weights/{}_cv/weights_fold_{}_from_{}.hdf5".format(args['dataset'], fold_nr, N_FOLDS)
        train_data_frame = pd.DataFrame(data={'filename': x[train], 'class': y[train]})
        test_data_frame = pd.DataFrame(data={'filename': x[test], 'class': y[test]})

        trn_flow, val_flow = get_training_and_validation_flow(train_data_frame, directory)
        tst_flow = get_testing_flow(test_data_frame, directory)

        if args['mode'] == Mode.train:
            model = create_model()
            model.fit_generator(generator=trn_flow, steps_per_epoch=(trn_flow.n // BATCH_SIZE),
                                validation_data=val_flow, validation_steps=val_flow.n,
                                callbacks=get_callbacks(filepath), epochs=EPOCHS)
        else:
            model = load_model(filepath)

        y_pred_prob = model.predict_generator(generator=tst_flow, verbose=1, steps=tst_flow.n)
        y_test = tst_flow.classes

        y_pred_probs.extend(list(zip(y_pred_prob.flatten().tolist(), y_test)))
        y_pred = np.where(y_pred_prob > 0.5, 1, 0).flatten()

        precision, recall, f_score, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        accuracy = accuracy_score(y_test, y_pred)

        metrics_dict["precision"] += precision
        metrics_dict["recall"] += recall
        metrics_dict["f-score"] += f_score
        metrics_dict["accuracy"] += accuracy

        print("F-Score : {}".format(f_score))
        print("Accuracy: {}".format(accuracy))

        fold_nr += 1

    print("Mean F-Score ------------> {}".format(metrics_dict["f-score"] / N_FOLDS))
    print("Mean Precision ----------> {}".format(metrics_dict["precision"] / N_FOLDS))
    print("Mean Recall -------------> {}".format(metrics_dict["recall"] / N_FOLDS))
    print("Mean Accuracy -----------> {}".format(metrics_dict["accuracy"] / N_FOLDS))
    calculate_average_precision_ks(y_pred_probs)


def calculate_average_precision_ks(lst, ks=(50, 100, 250, 480)):
    results, score = sorted(lst, key=itemgetter(0), reverse=True), 0

    for k in ks:
        y_score, y_true = zip(*results[:k])
        y_score, y_true = np.asarray(y_score), np.asarray(y_true)
        average_precision = average_precision_score(y_true, y_score)
        score += average_precision
        print("Average Precision @ {:<3} -> {}".format(k, average_precision))

    print("Average Precision @ {} -> {}".format(ks, score / len(ks)))


# noinspection PyTypeChecker
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest="mode", choices=list(Mode), type=Mode.from_string, required=True)
    parser.add_argument("--dataset", dest="dataset", choices=list(Dataset), type=Dataset.from_string, required=True)
    parser.add_argument("--method", dest="method", choices=list(Method), type=Method.from_string, required=True)
    return vars(parser.parse_args())


def train_test_model(args):
    directory, info = get_train_dataset_info(args["dataset"])

    if args['method'] == Method.cross_validation:
        train_test_model_cv(directory, info, args)
    if args['method'] == Method.train_test_split:
        train_test_model_split(directory, info, args)


if __name__ == "__main__":
    initial_configs()
    train_test_model(parse_args())
