import pandas as pd

from helpers.arguments import Dataset


def get_train_dataset_info(selected):
    images_directory, labels = None, None

    if selected == Dataset.mediaeval2017:
        images_directory = "./datasets/MediaEval2017/Classification/development_set/devset_images"
        labels = "./datasets/MediaEval2017/Classification/development_set/devset_images_gt.csv"

    if selected == Dataset.european_floods:
        images_directory = "./datasets/EuropeanFlood2013/imgs_small"
        labels = "./datasets/EuropeanFlood2013/classification.csv"

    return images_directory, pd.read_csv(
        labels,
        names=["filename", "class"],
    )


def get_test_dataset_info(selected):
    images_directory, labels = None, None

    if selected == Dataset.mediaeval2017:
        images_directory = "./datasets/MediaEval2017/Classification/test_set/testset_images"
        labels = "./datasets/MediaEval2017/Classification/test_set/testset_images_gt.csv"

    if selected == Dataset.european_floods:
        print("There is no test split for this dataset.")
        exit(0)

    return images_directory, pd.read_csv(
        labels,
        names=["filename", "class"],
    )
