import os

import pandas as pd

from helpers.arguments import Dataset

MEDIA_EVAL_2017_TRAIN_SEVERITY_LABELS = "./datasets/FloodSeverity/dataset_train_mediaeval_2017.csv"
MEDIA_EVAL_2017_TRAIN_DIRECTORY = "./datasets/MediaEval2017/Classification/development_set/devset_images"
MEDIA_EVAL_2017_TRAIN_LABELS = "./datasets/MediaEval2017/Classification/development_set/devset_images_gt.csv"

MEDIA_EVAL_2017_TEST_SEVERITY_LABELS = "./datasets/FloodSeverity/dataset_test_mediaeval_2017.csv"
MEDIA_EVAL_2017_TEST_DIRECTORY = "./datasets/MediaEval2017/Classification/test_set/testset_images"
MEDIA_EVAL_2017_TEST_LABELS = "./datasets/MediaEval2017/Classification/test_set/testset_images_gt.csv"

MEDIA_EVAL_2018_TRAIN_SEVERITY_LABELS = "./datasets/FloodSeverity/dataset_train_mediaeval_2018.csv"
MEDIA_EVAL_2018_TRAIN_DIRECTORY = "./datasets/MediaEval2018/Classification/development_set/devset_images"
MEDIA_EVAL_2018_TRAIN_LABELS = "./datasets/MediaEval2018/Classification/development_set/devset_images_gt.csv"

MEDIA_EVAL_2018_TEST_SEVERITY_LABELS = "./datasets/FloodSeverity/dataset_test_mediaeval_2018.csv"
MEDIA_EVAL_2018_TEST_DIRECTORY = "./datasets/MediaEval2018/Classification/test_set/testset_images"
MEDIA_EVAL_2018_TEST_LABELS = "./datasets/MediaEval2018/Classification/test_set/testset_images_gt.csv"

EUROPEAN_FLOOD_2013_SEVERITY_LABELS = "./datasets/FloodSeverity/dataset_european_flood_2013.csv"
EUROPEAN_FLOOD_2013_DIRECTORY = "./datasets/EuropeanFlood2013/imgs_small"
EUROPEAN_FLOOD_2013_LABELS = "./datasets/EuropeanFlood2013/classification.csv"


def join_full_path(folder, labels):
    directory = os.path.abspath(folder) + "/"
    labels = pd.read_csv(labels, names=['filename', 'class'], dtype=str)

    for index, row in labels.iterrows():
        for el in os.listdir(directory):
            if el.split(".")[0] == row[0]:
                row[0] = directory + el
                labels.iloc[index] = row
                break

    return labels[~labels.filename.str.endswith(".gif")]


def get_train_dataset_info(selected):
    if selected == Dataset.mediaeval2017:
        return join_full_path(MEDIA_EVAL_2017_TRAIN_DIRECTORY, MEDIA_EVAL_2017_TRAIN_LABELS)

    if selected == Dataset.european_floods:
        return join_full_path(EUROPEAN_FLOOD_2013_DIRECTORY, EUROPEAN_FLOOD_2013_LABELS)

    if selected == Dataset.both:
        european_floods = join_full_path(EUROPEAN_FLOOD_2013_DIRECTORY, EUROPEAN_FLOOD_2013_LABELS)
        media_eval_2017 = join_full_path(MEDIA_EVAL_2017_TRAIN_DIRECTORY, MEDIA_EVAL_2017_TRAIN_LABELS)
        media_eval_2018_train = join_full_path(MEDIA_EVAL_2018_TRAIN_DIRECTORY, MEDIA_EVAL_2018_TRAIN_LABELS)
        media_eval_2018_test = join_full_path(MEDIA_EVAL_2018_TEST_DIRECTORY, MEDIA_EVAL_2018_TEST_LABELS)
        result = european_floods.append(media_eval_2017, ignore_index=True)
        result = result.append(media_eval_2018_train, ignore_index=True).append(media_eval_2018_test, ignore_index=True)
        return result.drop(result[result['class'] == str(4)].index).reset_index(drop=True)

    if selected == Dataset.flood_severity_4_classes or selected == Dataset.flood_severity_3_classes:
        media_eval_test = join_full_path(MEDIA_EVAL_2017_TEST_DIRECTORY, MEDIA_EVAL_2017_TEST_SEVERITY_LABELS)
        media_eval_train = join_full_path(MEDIA_EVAL_2017_TRAIN_DIRECTORY, MEDIA_EVAL_2017_TRAIN_SEVERITY_LABELS)
        european_floods = join_full_path(EUROPEAN_FLOOD_2013_DIRECTORY, EUROPEAN_FLOOD_2013_SEVERITY_LABELS)
        result = media_eval_test.append(media_eval_train, ignore_index=True).append(european_floods, ignore_index=True)
        result = result.drop(result[result['class'] == str(4)].index).reset_index(drop=True)

        if selected == Dataset.flood_severity_3_classes:
            result = result.replace({'class': {'3': '2'}})

        return result

    if selected == Dataset.flood_severity_european_floods:
        result = join_full_path(EUROPEAN_FLOOD_2013_DIRECTORY, EUROPEAN_FLOOD_2013_SEVERITY_LABELS)
        result = result.drop(result[result['class'] == str(4)].index).reset_index(drop=True)
        return result.replace({'class': {'3': '2'}})


def get_test_dataset_info(selected):
    if selected not in [Dataset.flood_severity_3_classes, Dataset.flood_severity_4_classes,
                        Dataset.flood_severity_european_floods]:
        return join_full_path(MEDIA_EVAL_2017_TEST_DIRECTORY, MEDIA_EVAL_2017_TEST_LABELS)
    raise ValueError("There is not test split for flood severity dataset.")
