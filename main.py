import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import metrics
from keras.applications.densenet import DenseNet121
from keras.backend import set_session
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras_preprocessing.image import ImageDataGenerator
from numpy.random import seed
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow import set_random_seed

RANDOM_SEED = 40

BATCH_SIZE = 10
N_FOLDS = 5
EPOCHS = 50


def initial_configs():
    seed(RANDOM_SEED)
    set_random_seed(RANDOM_SEED)
    np.set_printoptions(threshold=np.inf)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)


def create_model():
    base_model = DenseNet121(include_top=False, weights='imagenet')
    optimizer = Adam(lr=1e-4, amsgrad=True)

    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[metrics.binary_accuracy])

    return model


def freeze_first_layers(model):
    for layer in model.layers[:-4]:
        layer.trainable = False

    optimizer = Adam(lr=1e-3, amsgrad=True)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[metrics.binary_accuracy])
    return model


def unfreeze_first_layers(model):
    # for layer in model.layers[:137]:
    #     layer.trainable = False
    #
    # for layer in model.layers[137:]:
    #     layer.trainable = True

    for layer in model.layers:
        layer.trainable = True

    optimizer = SGD(lr=1e-3, momentum=0.9)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[metrics.binary_accuracy])
    return model


def get_callbacks():
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1, mode='auto', verbose=1)

    return [early_stopping, reduce_lr]


def get_training_and_validation_flow(df, directory):
    image_generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.15)

    trn_flow = image_generator.flow_from_dataframe(dataframe=df, directory=directory,
                                                   subset="training", class_mode="binary",
                                                   shuffle=True, has_ext=False, batch_size=BATCH_SIZE)

    val_flow = image_generator.flow_from_dataframe(dataframe=df, directory=directory,
                                                   subset="validation", class_mode="binary",
                                                   shuffle=True, has_ext=False, batch_size=1)

    return trn_flow, val_flow


def get_testing_flow(df, directory):
    image_generator = ImageDataGenerator(rescale=1. / 255)

    tst_flow = image_generator.flow_from_dataframe(dataframe=df, directory=directory, shuffle=False,
                                                   class_mode="binary", has_ext=False, batch_size=1)

    return tst_flow


def train_test_model(directory, info):
    metrics_dict = {"precision": 0, "recall": 0, "f-score": 0, "accuracy": 0}
    x, y = info.iloc[:, 0].values, info.iloc[:, 1].values
    k_fold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for train, test in k_fold.split(x, y):

        train_data_frame = pd.DataFrame(data={'filename': x[train], 'class': y[train]})
        test_data_frame = pd.DataFrame(data={'filename': x[test], 'class': y[test]})

        trn_flow, val_flow = get_training_and_validation_flow(train_data_frame, directory)
        tst_flow = get_testing_flow(test_data_frame, directory)

        model = freeze_first_layers(create_model())
        model.fit_generator(generator=trn_flow, steps_per_epoch=(trn_flow.n // BATCH_SIZE), epochs=20)

        model = unfreeze_first_layers(model)
        model.fit_generator(generator=trn_flow, steps_per_epoch=(trn_flow.n // BATCH_SIZE),
                            validation_data=val_flow, validation_steps=val_flow.n,
                            callbacks=get_callbacks(), epochs=EPOCHS)

        y_pred = model.predict_generator(generator=tst_flow, steps=tst_flow.n)
        y_pred = np.where(y_pred > 0.5, 1, 0).flatten()
        y_test = tst_flow.classes

        precision, recall, f_score, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        accuracy = accuracy_score(y_test, y_pred)

        metrics_dict["precision"] += precision
        metrics_dict["recall"] += recall
        metrics_dict["f-score"] += f_score
        metrics_dict["accuracy"] += accuracy

        print("F-Score : {}".format(f_score))
        print("Accuracy: {}".format(accuracy))

    print("Mean F-Score  : {}".format(metrics_dict["f-score"] / N_FOLDS))
    print("Mean Precision: {}".format(metrics_dict["precision"] / N_FOLDS))
    print("Mean Recall   : {}".format(metrics_dict["recall"] / N_FOLDS))
    print("Mean Accuracy : {}".format(metrics_dict["accuracy"] / N_FOLDS))


def main():
    directory = "./datasets/MediaEval2017/Classification/development_set/devset_images"

    info = pd.read_csv(
        'datasets/MediaEval2017/Classification/development_set/devset_images_gt.csv',
        names=["filename", "class"],
    )

    train_test_model(directory, info)


if __name__ == "__main__":
    initial_configs()
    main()
