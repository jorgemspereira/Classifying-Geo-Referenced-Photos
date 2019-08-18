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

FLOOD_HEIGHTS_LABELS = "./datasets/FloodHeight/flood_height.csv"


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


def get_full_name(name, path):
    for el in os.listdir(path):
        if el.split(".")[0] == name:
            return path + el


def flood_height_aux(labels, split):
    directory_mediaeval_train = os.path.abspath(MEDIA_EVAL_2017_TRAIN_DIRECTORY) + "/"
    directory_mediaeval_test = os.path.abspath(MEDIA_EVAL_2017_TEST_DIRECTORY) + "/"
    directory_european_floods = os.path.abspath(EUROPEAN_FLOOD_2013_DIRECTORY) + "/"
    labels = pd.read_csv(labels, dtype=str)

    for index, row in labels.iterrows():
        if row['font'] == "mediaeval_2017_test":
            row['filename'] = get_full_name(row['filename'], directory_mediaeval_test)
        elif row['font'] == "mediaeval_2017_train":
            row['filename'] = get_full_name(row['filename'], directory_mediaeval_train)
        elif row['font'] == "european_floods_2013":
            row['filename'] = get_full_name(row['filename'], directory_european_floods)
        labels.iloc[index] = row

    if split == "train":
        media_eval_2017 = join_full_path(MEDIA_EVAL_2017_TRAIN_DIRECTORY, MEDIA_EVAL_2017_TRAIN_LABELS)
        european_floods = join_full_path(EUROPEAN_FLOOD_2013_DIRECTORY, EUROPEAN_FLOOD_2013_LABELS)
        result = media_eval_2017.append(european_floods, ignore_index=True)
        result = result.drop(result[result['class'] != str(0)].index).reset_index(drop=True)
        result['height'] = 0

        first_labels = labels[~labels.font.str.startswith("mediaeval_2017_test")]
        labels = first_labels.append(first_labels, ignore_index=True)
        labels = labels.append(first_labels, ignore_index=True)
        labels = labels.append(result, ignore_index=True, sort=False)

    elif split == "test":
        labels = labels[labels.font.str.startswith("mediaeval_2017_test")]
        # TODO (perguntar ao prof. Bruno se e para avaliar com as que nao tem cheia)

    else:
        raise ValueError("Split should be equal to train or test.")

    labels = labels.drop(['font'], axis=1).reset_index(drop=True)
    return labels[~labels.filename.str.endswith(".gif")]


def get_train_dataset_info(selected):
    if selected == Dataset.mediaeval2017:
        return join_full_path(MEDIA_EVAL_2017_TRAIN_DIRECTORY, MEDIA_EVAL_2017_TRAIN_LABELS)

    if selected == Dataset.mediaeval2018:
        train = join_full_path(MEDIA_EVAL_2018_TRAIN_DIRECTORY, MEDIA_EVAL_2018_TRAIN_LABELS)
        test = join_full_path(MEDIA_EVAL_2018_TEST_DIRECTORY, MEDIA_EVAL_2018_TEST_LABELS)
        result = train.append(test, ignore_index=True)
        return result.drop(result[result['class'] == str(4)].index).reset_index(drop=True)

    if selected == Dataset.european_floods:
        return join_full_path(EUROPEAN_FLOOD_2013_DIRECTORY, EUROPEAN_FLOOD_2013_LABELS)

    if selected == Dataset.all:
        european_floods = join_full_path(EUROPEAN_FLOOD_2013_DIRECTORY, EUROPEAN_FLOOD_2013_LABELS)
        media_eval_2017 = join_full_path(MEDIA_EVAL_2017_TRAIN_DIRECTORY, MEDIA_EVAL_2017_TRAIN_LABELS)
        media_eval_2018_train = join_full_path(MEDIA_EVAL_2018_TRAIN_DIRECTORY, MEDIA_EVAL_2018_TRAIN_LABELS)
        media_eval_2018_test = join_full_path(MEDIA_EVAL_2018_TEST_DIRECTORY, MEDIA_EVAL_2018_TEST_LABELS)
        result = european_floods.append(media_eval_2017, ignore_index=True)
        result = result.append(media_eval_2018_train, ignore_index=True).append(media_eval_2018_test, ignore_index=True)
        return result.drop(result[result['class'] == str(4)].index).reset_index(drop=True)

    if selected == Dataset.flood_severity_4_classes or selected == Dataset.flood_severity_3_classes:
        media_eval_test_2017 = join_full_path(MEDIA_EVAL_2017_TEST_DIRECTORY, MEDIA_EVAL_2017_TEST_SEVERITY_LABELS)
        media_eval_train_2017 = join_full_path(MEDIA_EVAL_2017_TRAIN_DIRECTORY, MEDIA_EVAL_2017_TRAIN_SEVERITY_LABELS)
        media_eval_test_2018 = join_full_path(MEDIA_EVAL_2018_TEST_DIRECTORY, MEDIA_EVAL_2018_TEST_SEVERITY_LABELS)
        media_eval_train_2018 = join_full_path(MEDIA_EVAL_2018_TRAIN_DIRECTORY, MEDIA_EVAL_2018_TRAIN_SEVERITY_LABELS)
        european_floods = join_full_path(EUROPEAN_FLOOD_2013_DIRECTORY, EUROPEAN_FLOOD_2013_SEVERITY_LABELS)
        result = media_eval_test_2017.append(media_eval_train_2017, ignore_index=True)
        result = result.append(media_eval_train_2018, ignore_index=True)
        result = result.append(media_eval_test_2018, ignore_index=True)
        result = result.append(european_floods, ignore_index=True)
        result = result.drop(result[result['class'] == str(4)].index).reset_index(drop=True)

        if selected == Dataset.flood_severity_3_classes:
            result = result.replace({'class': {'3': '2'}})

        return result

    if selected == Dataset.flood_severity_european_floods:
        result = join_full_path(EUROPEAN_FLOOD_2013_DIRECTORY, EUROPEAN_FLOOD_2013_SEVERITY_LABELS)
        result = result.drop(result[result['class'] == str(4)].index).reset_index(drop=True)
        return result.replace({'class': {'3': '2'}})

    if selected == Dataset.flood_heights:
        return flood_height_aux(FLOOD_HEIGHTS_LABELS, split="train")


def get_test_dataset_info(selected):
    if selected in [Dataset.mediaeval2017, Dataset.mediaeval2018, Dataset.european_floods, Dataset.all]:
        return join_full_path(MEDIA_EVAL_2017_TEST_DIRECTORY, MEDIA_EVAL_2017_TEST_LABELS)
    if selected == Dataset.flood_heights:
        return flood_height_aux(FLOOD_HEIGHTS_LABELS, split="test")
    raise ValueError("There is not test split for flood severity dataset.")
