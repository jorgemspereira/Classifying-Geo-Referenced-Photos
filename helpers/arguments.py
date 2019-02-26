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
    european_floods = 2

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
