import os

import pandas as pd

from helpers.arguments import Dataset

MEDIA_EVAL_2017_TRAIN_SEVERITY_LABELS = "./datasets/FloodSeverity/dataset_train_mediaeval_2017.csv"
MEDIA_EVAL_2017_TRAIN_DIRECTORY = "./datasets/MediaEval2017/Classification/development_set/devset_images"
MEDIA_EVAL_2017_TRAIN_LABELS = "./datasets/MediaEval2017/Classification/development_set/devset_images_gt.csv"

MEDIA_EVAL_2017_TEST_SEVERITY_LABELS = "./datasets/FloodSeverity/dataset_test_mediaeval_2017.csv"
MEDIA_EVAL_2017_TEST_DIRECTORY = "./datasets/MediaEval2017/Classification/test_set/testset_images"
MEDIA_EVAL_2017_TEST_LABELS = "./datasets/MediaEval2017/Classification/test_set/testset_images_gt.csv"

EUROPEAN_FLOOD_2013_SEVERITY_LABELS = "./datasets/FloodSeverity/dataset_european_flood_2013.csv"
EUROPEAN_FLOOD_2013_DIRECTORY = "./datasets/EuropeanFlood2013/imgs_small"
EUROPEAN_FLOOD_2013_LABELS = "./datasets/EuropeanFlood2013/classification_2.csv"


def join_full_path(folder, labels):
    directory = os.path.abspath(folder) + "/"
    labels = pd.read_csv(labels, names=['filename', 'class'], dtype=str)

    for index, row in labels.iterrows():
        for el in os.listdir(directory):
            if el.split(".")[0] == row[0]:
                row[0] = directory + el
                labels.iloc[index] = row
                break

    return labels


def get_train_dataset_info(selected):
    if selected == Dataset.mediaeval2017:
        return join_full_path(MEDIA_EVAL_2017_TRAIN_DIRECTORY, MEDIA_EVAL_2017_TRAIN_LABELS)

    if selected == Dataset.european_floods:
        return join_full_path(EUROPEAN_FLOOD_2013_DIRECTORY, EUROPEAN_FLOOD_2013_LABELS)

    if selected == Dataset.both:
        european_floods = join_full_path(EUROPEAN_FLOOD_2013_DIRECTORY, EUROPEAN_FLOOD_2013_LABELS)
        media_eval = join_full_path(MEDIA_EVAL_2017_TRAIN_DIRECTORY, MEDIA_EVAL_2017_TRAIN_LABELS)
        return european_floods.append(media_eval, ignore_index=True)

    if selected == Dataset.flood_severity:
        media_eval_test = join_full_path(MEDIA_EVAL_2017_TEST_DIRECTORY, MEDIA_EVAL_2017_TEST_SEVERITY_LABELS)
        media_eval_train = join_full_path(MEDIA_EVAL_2017_TRAIN_DIRECTORY, MEDIA_EVAL_2017_TRAIN_SEVERITY_LABELS)
        european_floods = join_full_path(EUROPEAN_FLOOD_2013_DIRECTORY, EUROPEAN_FLOOD_2013_SEVERITY_LABELS)
        result = media_eval_test.append(media_eval_train, ignore_index=True).append(european_floods, ignore_index=True)
        return result.drop(result[result['class'] == str(4)].index).reset_index(drop=True)


def get_test_dataset_info(selected):
    if selected != Dataset.flood_severity:
        return join_full_path(MEDIA_EVAL_2017_TEST_DIRECTORY, MEDIA_EVAL_2017_TEST_LABELS)
    raise ValueError("There is not test split for flood severity dataset.")


def get_test_images_directory():
    return MEDIA_EVAL_2017_TEST_DIRECTORY
