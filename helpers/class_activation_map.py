import os

import cv2
import keras.backend as K
import numpy as np
from keras.applications.densenet import preprocess_input
from keras.engine.saving import load_model
from keras.preprocessing import image
from tqdm import tqdm

from helpers.arguments import Method
from helpers.dataset import get_test_images_directory


def visualize_class_activation_map(model, img_path, output_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    model.predict(x)

    last_conv_layer = model.get_layer("conv5_block32_2_conv")
    grads = K.gradients(model.output[:, 0], last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(conv_layer_output_value.shape[2]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heat_map = np.mean(conv_layer_output_value, axis=-1)
    heat_map = np.maximum(heat_map, 0)
    heat_map /= np.max(heat_map)

    img = cv2.imread(img_path)
    heat_map = cv2.resize(heat_map, (img.shape[1], img.shape[0]))
    heat_map = np.uint8(255 * heat_map)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    superimposed_img = heat_map * .8 + img
    cv2.imwrite(output_path, superimposed_img)


def draw_class_activation_map(args):
    if args['method'] != Method.train_test_split:
        return

    output_directory = "./class_activation_maps/trained_by_{}".format(args["dataset"])
    input_directory = get_test_images_directory()

    model_path = "./weights/{}_split/weights.hdf5".format(args["dataset"])
    model = load_model(model_path)

    for photo in tqdm(os.listdir(input_directory)):
        input_path = "{}/{}".format(input_directory, photo)
        output_path = "{}/{}".format(output_directory, photo).replace(".jpg", ".bmp")
        visualize_class_activation_map(model, input_path, output_path)
