from keras import activations, losses, metrics, Model
from keras.applications import DenseNet201
from keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.engine.saving import load_model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import Adam

from callbacks.CyclicLR import CyclicLR
from helpers.arguments import Mode
from helpers.focal_loss import categorical_focal_loss


def create_model(number_classes):
    base_model = DenseNet201(include_top=False, weights='imagenet')
    optimizer = Adam()

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    if number_classes == 2:
        print("Binary model.")
        predictions = Dense(1, activation=activations.sigmoid)(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=optimizer, loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
    else:
        print("{} number of classes.".format(number_classes))
        predictions = Dense(number_classes, activation=activations.softmax)(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=optimizer, metrics=[metrics.categorical_accuracy],
                      loss=[categorical_focal_loss(gamma=2., alpha=.25)])

    return model


def get_callbacks(filepath):
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='auto', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)
    cyclic_lr = CyclicLR(base_lr=1e-5, step_size=2000., max_lr=1e-4, mode='triangular2')
    terminate_nan = TerminateOnNaN()

    return [early_stopping, checkpoint, cyclic_lr, terminate_nan]


def train_or_load_model(args, trn_flow, val_flow, batch_size, filepath, epochs, is_binary):
    if args['mode'] == Mode.train:
        model = create_model(number_classes=len(set(trn_flow.classes)))
        # class_weights = None if is_binary else \
        #    compute_class_weight('balanced', np.unique(trn_flow.classes), trn_flow.classes)
        model.fit_generator(generator=trn_flow, steps_per_epoch=(trn_flow.n // batch_size),
                            validation_data=val_flow, validation_steps=val_flow.n,
                            callbacks=get_callbacks(filepath), epochs=epochs)  # , class_weight=class_weights)
    else:
        model = load_model(filepath)

    return model
