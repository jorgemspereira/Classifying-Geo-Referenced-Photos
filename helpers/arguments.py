from enum import Enum


class Mode(Enum):
    train = 1
    load = 2

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Mode[s]
        except KeyError:
            raise ValueError()


class Dataset(Enum):
    mediaeval2017 = 1
    mediaeval2018 = 2
    european_floods = 3
    all = 4
    flood_severity_4_classes = 5
    flood_severity_3_classes = 6
    flood_severity_european_floods = 7
    flood_heights = 8

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Dataset[s]
        except KeyError:
            raise ValueError()


class Method(Enum):
    cross_validation = 1
    train_test_split = 2

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Method[s]
        except KeyError:
            raise ValueError()


class Model(Enum):
    dense_net = 1
    attention_guided = 2
    efficient_net = 3

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Model[s]
        except KeyError:
            raise ValueError()
