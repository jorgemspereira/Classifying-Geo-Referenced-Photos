import numpy as np
from keras import activations, losses, metrics, Model
from keras.applications import DenseNet201
from keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.engine.saving import load_model
from keras.layers import GlobalAveragePooling2D, Dense, concatenate, Dropout
from keras.models import clone_model
from keras.optimizers import Adam

from callbacks.CyclicLR import CyclicLR
from helpers.arguments import Mode
from helpers.focal_loss import categorical_class_balanced_focal_loss


def count(lst):
    result = [0] * len(set(lst))
    for el in lst:
        result[el] += 1
    return result


def create_model(trn_flow, number_classes):
    base_model = DenseNet201(include_top=False, weights='imagenet')
    optimizer = Adam(lr=1e-5)

    x = base_model.output
    x = GlobalAveragePooling2D(name="global_average_pooling2d_layer")(x)

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
                      loss=[categorical_class_balanced_focal_loss(count(trn_flow.classes), beta=0.9, gamma=2.)])

    return model


def create_fused_model(number_classes, training_classes, branches_paths):
    optimizer = Adam(lr=1e-5)

    global_branch_model = branches_paths["model_global"]
    local_branch_model = branches_paths["model_local"]

    for layer in global_branch_model.layers:
        layer.name = layer.name + "_m1"
        layer.trainable = False

    for layer in local_branch_model.layers:
        layer.trainable = False

    last_avgpooling_global = global_branch_model.get_layer("global_average_pooling2d_layer_m1")
    last_avgpooling_local = local_branch_model.get_layer("global_average_pooling2d_layer")
    output = concatenate([last_avgpooling_global.output, last_avgpooling_local.output])
    output = Dropout(0.5)(output)

    if number_classes == 2:
        print("Binary model.")
        predictions = Dense(1, activation=activations.sigmoid)(output)
        model = Model(inputs=[global_branch_model.input, local_branch_model.input], outputs=predictions)
        model.compile(optimizer=optimizer, loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
    else:
        print("{} number of classes.".format(number_classes))
        predictions = Dense(number_classes, activation=activations.softmax)(output)
        model = Model(inputs=[global_branch_model.input, local_branch_model.input], outputs=predictions)
        model.compile(optimizer=optimizer, metrics=[metrics.categorical_accuracy],
                      loss=[categorical_class_balanced_focal_loss(count(training_classes), beta=0.9, gamma=2.)])

    model.layers[-1].set_weights([np.concatenate((global_branch_model.layers[-1].get_weights()[0],
                                                  local_branch_model.layers[-1].get_weights()[0])),
                                  np.mean(np.array([global_branch_model.layers[-1].get_weights()[1],
                                                    local_branch_model.layers[-1].get_weights()[1]]), axis=0)])

    return model


def get_callbacks(filepath):
    return [
        ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='auto', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True),
        CyclicLR(base_lr=1e-5, step_size=1000., max_lr=1e-4, mode='triangular2'),
        TerminateOnNaN()
    ]


def train_or_load_model(args, trn_flow, val_flow, filepath, training_classes, training_examples,
                        validation_examples, branches_models=None, pre_trained_model=None):

    if args['mode'] == Mode.train:
        if pre_trained_model is not None:
            model = clone_model(pre_trained_model)
            model.set_weights(pre_trained_model.get_weights())
            model.compile(optimizer=Adam(lr=1e-4), metrics=[metrics.categorical_accuracy],
                          loss=[categorical_class_balanced_focal_loss(count(training_classes), beta=0.9, gamma=2.)])

        elif branches_models is not None:
            model = create_fused_model(len(set(training_classes)), training_classes, branches_models)
        else:
            model = create_model(trn_flow, number_classes=len(set(training_classes)))

        model.fit_generator(generator=trn_flow, steps_per_epoch=(training_examples // args['batch_size']),
                            validation_data=val_flow, validation_steps=validation_examples,
                            callbacks=get_callbacks(filepath), epochs=args['epochs'])
    else:
        # FIXME
        custom_object = {'categorical_class_balanced_focal_loss_fixed': lambda y_true, y_pred: y_pred,
                         'categorical_class_balanced_focal_loss': lambda y_true, y_pred: y_pred}

        model = load_model(filepath, custom_objects=custom_object)

    return model
