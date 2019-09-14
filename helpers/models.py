import numpy as np
from efficientnet import EfficientNetB3
from keras import activations, losses, metrics, Model
from keras import backend as K
from keras.applications import DenseNet201
from keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.engine.saving import load_model
from keras.layers import GlobalAveragePooling2D, Dense, concatenate, Dropout
from keras.models import clone_model
from keras.optimizers import Adam

from callbacks.CyclicLR import CyclicLR
from helpers import arguments
from helpers.arguments import Mode, Dataset
from helpers.custom_layers import OutputLayer


def count(lst):
    result = [0] * len(set(lst))
    for el in lst:
        result[el] += 1
    return result


def set_weights(model, model_global, model_local, layer_index_from, layer_name_to):
    model.get_layer(layer_name_to).set_weights([
        np.concatenate((
            model_global.layers[layer_index_from].get_weights()[0],
            model_local.layers[layer_index_from].get_weights()[0])),
        np.mean(np.array([
            model_global.layers[layer_index_from].get_weights()[1],
            model_local.layers[layer_index_from].get_weights()[1]]), axis=0)])

    return model


def custom_activation_more_1m(x):
    return K.clip(K.relu(x), 1, max_value=None)


def custom_activation_less_1m(x):
    return K.clip(K.relu(x), 0, 1)


def create_model(args, branch):
    if args['model'] == arguments.Model.dense_net or args['model'] == arguments.Model.attention_guided:
        base_model = DenseNet201(include_top=False, weights='imagenet')
    else:
        base_model = EfficientNetB3(include_top=False, weights='imagenet')
    optz = Adam(lr=1e-5)

    x = base_model.output
    x = GlobalAveragePooling2D(name="global_average_pooling2d_layer")(x)

    if args['dataset'] != Dataset.flood_heights:

        if args['is_binary']:
            print("Binary model.")
            predictions = Dense(1, activation=activations.sigmoid)(x)
            model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(optimizer=optz, loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

        else:
            print("3 number of classes.")
            predictions = Dense(3, activation=activations.softmax)(x)
            model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(optimizer=optz, metrics=[metrics.categorical_accuracy], loss=losses.categorical_crossentropy)

    else:
        print("Regression model.")
        base_model = load_model("/home/jpereira/Tests/weights/" +
                                "flood_severity_3_classes_attention_guided_{}_branch_cv/".format(branch) +
                                "weights_fold_1_from_10.hdf5")

        output_less_1_m = Dense(1, activation=custom_activation_less_1m, name="less_1m")(base_model.layers[-2].output)
        output_more_1_m = Dense(1, activation=custom_activation_more_1m, name="more_1m")(base_model.layers[-2].output)
        predictions = OutputLayer()([base_model.layers[-1].output, output_less_1_m, output_more_1_m])

        model = Model(inputs=base_model.input, outputs=[base_model.layers[-1].output, predictions])
        model.compile(optimizer=optz, loss=[losses.categorical_crossentropy, losses.mean_squared_error])

    return model


def create_fused_model(args, branches_paths):
    optz = Adam(lr=1e-5)

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

    if args['dataset'] != Dataset.flood_heights:

        if args['is_binary']:
            print("Binary model.")
            predictions = Dense(1, activation=activations.sigmoid)(output)
            model = Model(inputs=[global_branch_model.input, local_branch_model.input], outputs=predictions)
            model.compile(optimizer=optz, loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

        else:
            print("3 number of classes.")
            predictions = Dense(3, activation=activations.softmax)(output)
            model = Model(inputs=[global_branch_model.input, local_branch_model.input], outputs=predictions)
            model.compile(optimizer=optz, metrics=[metrics.categorical_accuracy], loss=losses.categorical_crossentropy)

    else:
        print("Regression model.")
        pred_class = Dense(3, activation=activations.softmax, name="class_output")(output)
        output_less_1_m = Dense(1, activation=custom_activation_less_1m, name="less_1m")(output)
        output_more_1_m = Dense(1, activation=custom_activation_more_1m, name="more_1m")(output)
        predictions = OutputLayer()([pred_class, output_less_1_m, output_more_1_m])

        model = Model(inputs=[global_branch_model.input, local_branch_model.input], outputs=[pred_class, predictions])
        model.compile(optimizer=optz, loss=[losses.categorical_crossentropy, losses.mean_squared_error])

    if args['dataset'] != Dataset.flood_heights:
        model.layers[-1].set_weights([np.concatenate((global_branch_model.layers[-1].get_weights()[0],
                                                      local_branch_model.layers[-1].get_weights()[0])),
                                      np.mean(np.array([global_branch_model.layers[-1].get_weights()[1],
                                                        local_branch_model.layers[-1].get_weights()[1]]), axis=0)])
    else:

        model = set_weights(model, global_branch_model, local_branch_model, -4, "class_output")
        model = set_weights(model, global_branch_model, local_branch_model, -3, "less_1m")
        model = set_weights(model, global_branch_model, local_branch_model, -2, "more_1m")

    return model


def get_callbacks(filepath):
    return [
        ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='auto', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True),
        CyclicLR(base_lr=1e-5, step_size=1000., max_lr=1e-4, mode='triangular2'),
        TerminateOnNaN()
    ]


def train_or_load_model(args, trn_flow, val_flow, filepath, training_examples, validation_examples,
                        branches_models=None, branch=None, pre_trained_model=None):

    if args['mode'] == Mode.train:
        if pre_trained_model is not None:
            optz = Adam(lr=1e-4)
            model = clone_model(pre_trained_model)
            model.set_weights(pre_trained_model.get_weights())
            model.compile(optimizer=optz, metrics=[metrics.categorical_accuracy], loss=losses.categorical_crossentropy)

        elif branches_models is not None:
            model = create_fused_model(args, branches_models)

        else:
            model = create_model(args, branch)

        model.fit_generator(generator=trn_flow, steps_per_epoch=(training_examples // args['batch_size']),
                            validation_data=val_flow, validation_steps=validation_examples,
                            callbacks=get_callbacks(filepath), epochs=args['epochs'])
    else:
        custom_object = {'OutputLayer': OutputLayer,
                         'custom_activation_more_1m': custom_activation_more_1m,
                         'custom_activation_less_1m': custom_activation_less_1m}
        model = load_model(filepath, custom_objects=custom_object)

    return model
