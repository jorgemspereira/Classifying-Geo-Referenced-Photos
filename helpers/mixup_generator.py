import numpy as np


class MixupImageDataGenerator:
    def __init__(self, generator, dataframe, batch_size, class_mode, seed, target_size=None, shuffle=True, alpha=0.2):
        """Constructor for mixup image data generator.

        Arguments:
            generator {object} -- An instance of Keras ImageDataGenerator.
            dataframe {object} -- Pandas dataframe.
            batch_size {int} -- Batch size.
            target_size {tuple} -- Tuple height x width in pixels.
            class_mode {string} -- Class mode, either "categorical" or "binary".
            seed (int) -- Random seed.

        Keyword Arguments:
            shuffle (bool) -- Boolean to say if it is necessary to shuffle the data.
            alpha {float} -- Mixup beta distribution alpha parameter. (default: {0.2})
        """

        self.batch_index = 0
        self.batch_size = batch_size
        self.alpha = alpha

        # First iterator yielding tuples of (x, y)
        self.generator1 = generator.flow_from_dataframe(dataframe=dataframe,
                                                        directory=None,
                                                        target_size=target_size,
                                                        shuffle=shuffle,
                                                        class_mode=class_mode,
                                                        seed=seed,
                                                        batch_size=batch_size)

        # Second iterator yielding tuples of (x, y)
        self.generator2 = generator.flow_from_dataframe(dataframe=dataframe,
                                                        directory=None,
                                                        target_size=target_size,
                                                        shuffle=shuffle,
                                                        class_mode=class_mode,
                                                        batch_size=batch_size)

        # Number of images across all classes in image directory.
        self.class_indices = self.generator1.class_indices
        self.classes = self.generator1.classes
        self.n = self.generator1.samples

    def reset_index(self):
        """Reset the generator indexes array.
        """
        self.generator1.on_epoch_end()
        self.generator2.on_epoch_end()

    def on_epoch_end(self):
        self.reset_index()

    def reset(self):
        self.batch_index = 0

    def __len__(self):
        # round up
        return (self.n + self.batch_size - 1) // self.batch_size

    def get_steps_per_epoch(self):
        """Get number of steps per epoch based on batch size and
        number of images.

        Returns:
            int -- steps per epoch.
        """

        return self.n // self.batch_size

    def __next__(self):
        """Get next batch input/output pair.

        Returns:
            tuple -- batch of input/output pair, (inputs, outputs).
        """

        if self.batch_index == 0:
            self.reset_index()

        current_index = (self.batch_index * self.batch_size) % self.n
        if self.n > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        # random sample the lambda value from beta distribution.
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)

        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        # Get a pair of inputs and outputs from two iterators.
        X1, y1 = self.generator1.next()
        X2, y2 = self.generator2.next()

        if X1.shape[0] != self.batch_size:
            self.on_epoch_end()

            X1, y1 = self.generator1.next()
            X2, y2 = self.generator2.next()

        # Perform the mixup.
        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)

        return X, y

    def __iter__(self):
        while True:
            yield next(self)
