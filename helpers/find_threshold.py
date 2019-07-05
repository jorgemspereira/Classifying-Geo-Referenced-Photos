import json

import cv2
import numpy as np
import pandas as pd
from keras import backend as K
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


def intersection_over_union(box_complete, boxes):
    boxes_areas, intersect_areas = [], []

    for box in boxes:
        x_a = max(box[0], box_complete[0])
        y_a = max(box[1], box_complete[1])
        x_b = min(box[2], box_complete[2])
        y_b = min(box[3], box_complete[3])

        intersect_areas.append(max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1))
        boxes_areas.append((box[2] - box[0] + 1) * (box[3] - box[1] + 1))

    box_area_complete = (box_complete[2] - box_complete[0] + 1) * (box_complete[3] - box_complete[1] + 1)
    return sum(intersect_areas) / float(sum(boxes_areas) + box_area_complete - sum(intersect_areas))


def read_json():
    path_json = "./datasets/EuropeanFlood2013/bounding_boxes_depth.json"
    with open(path_json, encoding="utf-8") as f:
        data = json.load(f, encoding='utf-8')
    return data


def calculate_heat_map_aux(args, model, predicted_class, input_paths, output_paths):
    images, result = [], {}

    for img in input_paths:
        x = image.load_img(img, target_size=(args['image_size'], args['image_size']))
        x = image.img_to_array(x)
        x = np.divide(x, 255)
        x = np.expand_dims(x, axis=0)
        images.append(x)

    last_conv_layer = model.get_layer("conv5_block32_2_conv")
    output = model.output[:, predicted_class]

    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    for index, img_temp in enumerate(images):
        pooled_grads_value, conv_layer_output_value = iterate([img_temp])

        for i in range(conv_layer_output_value.shape[2]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heat_map = np.mean(conv_layer_output_value, axis=-1)

        if (heat_map < 0).all():
            avg = np.average(heat_map)
            heat_map += abs(avg)

        heat_map = np.maximum(heat_map, 0)
        heat_map /= np.max(heat_map)

        img = cv2.imread(input_paths[index])
        heat_map = cv2.resize(heat_map, (img.shape[1], img.shape[0]))
        heat_map = np.uint8(255 * heat_map)

        result[input_paths[index]] = heat_map

        # heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
        # superimposed_img = heat_map * .3 + img
        # cv2.imwrite(output_paths[index], superimposed_img)

    return result


def calculate_heat_map(args, model, data_frame):
    inputs, outputs, result = [], [], {}

    generator = ImageDataGenerator(rescale=1. / 255)
    flow = generator.flow_from_dataframe(dataframe=data_frame, directory=None,
                                         target_size=(args['image_size'], args['image_size']),
                                         shuffle=False, batch_size=1)
    predictions = model.predict_generator(flow, verbose=1, steps=flow.n)
    y_pred = np.argmax(predictions, axis=1)

    for index, row in data_frame.iterrows():
        input_path, photo = row[0], row[0].split("/")[-1].replace("jpg", "bmp")
        outputs.append("./maps/{}".format(photo))

        inputs.append(input_path)

    items = list(zip(y_pred, inputs, outputs))
    number_of_classes = len(set(y_pred))

    for c in range(number_of_classes):
        elements = [x for x in items if x[0] == c]
        inputs = [x[1] for x in elements]
        outputs = [x[2] for x in elements]
        result.update(calculate_heat_map_aux(args, model, c, inputs, outputs))

    return result


def crop_heat_map(args, heat_map, threshold=125):
    ret, mask = cv2.threshold(heat_map, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    max_area, max_index = 0, 0
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_index = index

    x, y, width, height = cv2.boundingRect(contours[max_index])
    aspect = width / float(height)

    ideal_width, ideal_height = args['image_size'], args['image_size']
    ideal_aspect = ideal_width / float(ideal_height)

    if aspect > ideal_aspect:
        new_width = int(ideal_aspect * height)
        offset = (width - new_width) / 2
        resize = (offset, 0, width - offset, height)
    else:
        new_height = int(width / ideal_aspect)
        offset = (height - new_height) / 2
        resize = (0, offset, width, height - offset)

    return int(x + resize[0]), int(y + resize[1]), int(x + resize[2]), int(y + resize[3])


def get_bounding_boxes(content):
    template_path = "./datasets/EuropeanFlood2013/imgs_small/{}.jpg"
    result = {}
    for key, value in content.items():
        rectangles = []

        img_path = template_path.format(key)
        img = cv2.imread(img_path)
        height_img, width_img, _ = img.shape

        for _, el in value.items():
            for bounding_box in el:
                height, width = int(bounding_box['height'] * height_img), int(bounding_box['width'] * width_img)
                left, top = int(bounding_box['left'] * width_img), int(bounding_box['top'] * height_img)
                rectangles.append((left, top, left + width, top + height))

        result[img_path] = rectangles

    return result


def build_data_frame(content):
    template_path = "./datasets/EuropeanFlood2013/imgs_small/{}.jpg"
    result = {}
    for key, value in content.items():
        result[template_path.format(key)] = "1"

    return pd.DataFrame(list(result.items()), columns=['filename', 'class'])


def calculate_scores(args, heat_maps, bounding_boxes):
    result = []

    for key in heat_maps.keys():
        max_result = 0
        max_threshold = 0

        for threshold in range(1, 250):
            attention = crop_heat_map(args, heat_maps[key], threshold)
            bboxes = bounding_boxes[key]
            metric = intersection_over_union(attention, bboxes)

            if metric >= max_result:
                max_result = metric
                max_threshold = threshold

        result.append(max_threshold)

    final_result = int(np.mean(result))
    print("Best threshold: {}".format(final_result))

    # for key in heat_maps.keys():
    #    img = cv2.imread(key)
    #    attention = crop_heat_map(heat_maps[key], final_result)
    #    cv2.rectangle(img, (attention[0], attention[1]), (attention[2], attention[3]), (0, 0, 0), 2)

    #    bboxes = bounding_boxes[key]
    #    for b in bboxes:
    #        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 2)
    #    cv2.imwrite("./bb/" + key.split("/")[-1], img)

    return final_result


def find_crop_threshold(args, model):
    content = read_json()
    bounding_boxes = get_bounding_boxes(content)

    data_frame = build_data_frame(content)
    heat_maps = calculate_heat_map(args, model, data_frame)
    return calculate_scores(args, heat_maps, bounding_boxes)
