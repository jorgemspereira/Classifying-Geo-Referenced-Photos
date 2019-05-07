import os

import cv2
import keras.backend as K
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from helpers.arguments import Mode


def check_path(filepath):
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    return filepath


def verify_probabilities(y_probs, train_flow):
    train_indices = train_flow.class_indices
    if train_indices['0'] != 0 and train_flow['1'] != 1:
        return np.array([1. - el for el in y_probs.flatten()])
    return y_probs.flatten()


def calculate_prediction(y_pred_prob, trn_flow, is_binary):
    if is_binary:
        y_pred_prob = verify_probabilities(y_pred_prob, trn_flow)
        y_pred = np.where(y_pred_prob > 0.5, 1, 0)
    else:
        predicted_class_indices = np.argmax(y_pred_prob, axis=1)
        labels = trn_flow.class_indices
        labels = dict((v, k) for k, v in labels.items())
        y_pred = [int(labels[k]) for k in predicted_class_indices]

    return y_pred


def crop_attention_map(img, heat_map, output_path, threshold=155):
    ret, mask = cv2.threshold(heat_map, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    max_area, max_index = 0, 0
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_index = index

    try:
        contour = contours[max_index]
        ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
        ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
        ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
        ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])
        cropped_image = img[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
    except IndexError:
        cropped_image = img

    output_path_cropped = output_path.replace("class_activation_maps", "cropped_class_activation_maps")
    cv2.imwrite(output_path_cropped, cropped_image)


def visualize_class_activation_map(model, predicted_class, is_binary, input_paths, output_paths, pb=None, crop=False):
    images = []
    for img in input_paths:
        x = image.load_img(img, target_size=(224, 224))
        x = image.img_to_array(x)
        x = np.divide(x, 255)
        x = np.expand_dims(x, axis=0)
        images.append(x)

    last_conv_layer = model.get_layer("conv5_block32_2_conv")
    output = model.output[:, 0] if is_binary else model.output[:, predicted_class]

    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    for index, img_temp in enumerate(images):
        pooled_grads_value, conv_layer_output_value = iterate([img_temp])

        for i in range(conv_layer_output_value.shape[2]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heat_map = np.mean(conv_layer_output_value, axis=-1)
        heat_map = np.maximum(heat_map, 0)
        heat_map /= np.max(heat_map)

        img = cv2.imread(input_paths[index])
        heat_map = cv2.resize(heat_map, (img.shape[1], img.shape[0]))
        heat_map = np.uint8(255 * heat_map)

        if crop:
            crop_attention_map(img, heat_map, output_paths[index])

        heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
        superimposed_img = heat_map * .3 + img
        cv2.imwrite(output_paths[index], superimposed_img)

        if pb is not None:
            pb.update(1)


def draw_class_activation_map(args, model, data_frame, train_flow):
    if not args['class_activation_map']:
        return

    output_directory = check_path("./class_activation_maps/trained_by_{}".format(args["dataset"]))
    output_directory = os.path.abspath(output_directory)

    outputs, inputs = [], []

    for index, row in data_frame.iterrows():
        input_path, photo = row[0], row[0].split("/")[-1]

        outputs.append("{}/{}".format(output_directory, photo).replace(".jpg", ".bmp"))
        inputs.append(input_path)

    print("Predicting images to generate class activation maps...")
    generator = ImageDataGenerator(rescale=1. / 255)
    flow = generator.flow_from_dataframe(dataframe=data_frame, directory=None,
                                         target_size=(224, 224), shuffle=False, batch_size=1)
    predictions = model.predict_generator(flow, verbose=1, steps=flow.n)
    y_pred = calculate_prediction(predictions, train_flow, args['is_binary'])
    print("Done.")

    print("Drawing class activation maps...")
    progress_bar = tqdm(total=len(inputs))

    items = list(zip(y_pred, inputs, outputs))
    number_of_classes = len(set(y_pred))
    for c in range(number_of_classes):
        elements = [x for x in items if x[0] == c]
        inputs = [x[1] for x in elements]
        outputs = [x[2] for x in elements]
        visualize_class_activation_map(model, c, args['is_binary'], inputs, outputs, pb=progress_bar)

    progress_bar.close()
    print("Done.")


def crop_and_draw_class_activation_map(args, model, data_frame, train_flow, fold_nr):
    check_path("./cropped_class_activation_maps/trained_by_{}_fold_{}".format(args["dataset"], fold_nr))

    output_directory = check_path("./class_activation_maps/trained_by_{}_fold_{}".format(args["dataset"], fold_nr))
    output_directory = os.path.abspath(output_directory)

    result_df = pd.DataFrame(columns=['filename', 'class'])
    outputs, inputs = [], []

    for index, row in data_frame.iterrows():
        input_path, photo = row[0], row[0].split("/")[-1]

        output_path = "{}/{}".format(output_directory, photo).replace(".jpg", ".bmp")
        output_path_cropped = output_path.replace("class_activation_maps", "cropped_class_activation_maps")

        inputs.append(input_path)
        outputs.append(output_path)

        result_df = result_df.append({"filename": output_path_cropped, "class": row['class']}, ignore_index=True)

    if args['mode'] == Mode.train:
        print("Predicting images to generate class activation maps...")
        generator = ImageDataGenerator(rescale=1. / 255)
        flow = generator.flow_from_dataframe(dataframe=data_frame, directory=None,
                                             target_size=(224, 224), shuffle=False, batch_size=1)
        predictions = model.predict_generator(flow, verbose=1, steps=flow.n)
        y_pred = calculate_prediction(predictions, train_flow, args['is_binary'])
        print("Done.")

        print("Drawing class activation maps...")
        progress_bar = tqdm(total=len(inputs))

        items = list(zip(y_pred, inputs, outputs))
        number_of_classes = len(set(y_pred))
        for c in range(number_of_classes):
            elements = [x for x in items if x[0] == c]
            inputs = [x[1] for x in elements]
            outputs = [x[2] for x in elements]
            visualize_class_activation_map(model, c, args['is_binary'], inputs, outputs, pb=progress_bar, crop=True)

        progress_bar.close()
        print("Done.")

    return result_df
