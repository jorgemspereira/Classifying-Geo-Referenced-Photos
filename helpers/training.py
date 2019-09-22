import os
from collections import defaultdict
from operator import itemgetter

import numpy as np
import pandas as pd
from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, \
    average_precision_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split

from helpers.arguments import Dataset
from helpers.class_activation_map import draw_class_activation_map, crop_and_draw_class_activation_map
from helpers.dataset import get_test_dataset_info, get_train_dataset_info
from helpers.mixup_generator import MixupImageDataGenerator
from helpers.models import train_or_load_model


def get_class_mode(args, class_mode):
    if class_mode is not None:
        return class_mode
    if args['dataset'] == Dataset.flood_heights:
        return "other"
    return "binary" if args['is_binary'] else "categorical"


def check_path(filepath):
    parent = os.path.dirname(filepath)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    return filepath


def create_flow(args, df, batch_size, shuffle=True, class_mode=None, data_augmentation=False, mix_up=False):
    if data_augmentation:
        generator = ImageDataGenerator(rescale=1. / 255,
                                       horizontal_flip=True,
                                       brightness_range=[0.8, 1.2])
    else:
        generator = ImageDataGenerator(rescale=1. / 255)

    if mix_up:
        flow = MixupImageDataGenerator(generator=generator, target_size=(args['image_size'], args['image_size']),
                                       dataframe=df, batch_size=batch_size,
                                       class_mode=get_class_mode(args, class_mode),
                                       seed=args['random_seed'], shuffle=shuffle)
    else:
        flow = generator.flow_from_dataframe(dataframe=df, directory=None, shuffle=shuffle,
                                             target_size=(args['image_size'], args['image_size']),
                                             class_mode=get_class_mode(args, class_mode),
                                             seed=args['random_seed'], batch_size=batch_size)
    flow.reset()
    return flow


def get_training_and_validation_flow(args, df, class_mode=None, split_size=0.10):
    random_seed = args['random_seed']
    x, y = df.iloc[:, 0].values, df.iloc[:, 1].values

    if args['dataset'] != Dataset.flood_heights:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=split_size,
                                                          stratify=y, random_state=random_seed)
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=split_size, random_state=random_seed)

    train_data_frame = pd.DataFrame(data={'filename': x_train, 'class': y_train})
    validation_data_frame = pd.DataFrame(data={'filename': x_val, 'class': y_val})

    train_flow = create_flow(args, train_data_frame, args['batch_size'], class_mode=class_mode,
                             data_augmentation=args['data_augmentation'], mix_up=args['mix_up'])
    validation_flow = create_flow(args, validation_data_frame, class_mode=class_mode, batch_size=1)

    return train_flow, validation_flow


def merge_input_generators(x1, x2):
    x1.reset()
    x2.reset()

    while True:
        x1i = x1.next()
        x2i = x2.next()
        yield [x1i[0], x2i[0]], x1i[1]


def merge_output_generators(x1, x2):
    x1.reset()
    x2.reset()

    while True:
        xli = x1.next()
        x2i = x2.next()
        yield xli[0], [xli[1], x2i[1]]


def merge_all_generators(global_1, local_2):
    global_1.reset()
    local_2.reset()

    while True:
        xli = global_1.next()
        x2i = local_2.next()
        yield [xli[0], x2i[0]], [xli[1], x2i[1]]


def verify_probabilities(y_probs, train_flow):
    train_indices = train_flow.class_indices
    if train_indices['0'] != 0 and train_flow['1'] != 1:
        return np.array([1. - el for el in y_probs.flatten()])
    return y_probs.flatten()


def calculate_prediction(args, y_pred_prob, trn_flow, tst_df):
    if args['dataset'] != Dataset.flood_heights:
        if args['is_binary']:
            y_pred_prob = verify_probabilities(y_pred_prob, trn_flow)
            y_pred = np.where(y_pred_prob > 0.5, 1, 0)
        else:
            predicted_class_indices = np.argmax(y_pred_prob, axis=1)
            labels = trn_flow.class_indices
            labels = dict((v, k) for k, v in labels.items())
            y_pred = [int(labels[k]) for k in predicted_class_indices]
    else:
        y_pred = y_pred_prob

    return y_pred_prob, y_pred, tst_df['class'].values


def calculate_accuracy_per_class(args, y_test, y_pred):
    if args['dataset'] != Dataset.flood_heights:
        cm = confusion_matrix(y_test.astype(int), y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Accuracy per class ------> {}".format(cm.diagonal()))


def calculate_accuracy_with_threshold(y_test, y_pred, threshold=0.1):
    correct, incorrect = 0, 0

    for index in range(0, len(y_test)):
        if y_pred[index] - threshold <= float(y_test[index]) <= y_pred[index] + threshold:
            correct += 1
        else:
            incorrect += 1

    return correct / (correct + incorrect)


def accuracy_precision_recall_fscore(args, y_test, y_pred, y_test2=None):
    if args['dataset'] != Dataset.flood_heights:
        accuracy = accuracy_score(y_test.astype(int), y_pred)
        result = {'accuracy': accuracy}

        if args['is_binary']:
            precision, recall, f_score, _ = precision_recall_fscore_support(y_test.astype(int), y_pred, average='binary')
            result.update({'precision': precision, 'recall': recall, 'f-score': f_score})

        else:
            mean_absolute = mean_absolute_error(y_test.astype(int), y_pred)
            precision_ma, recall_ma, f_score_ma, _ = precision_recall_fscore_support(y_test.astype(int), y_pred, average='macro')
            result.update({'precision_ma': precision_ma, 'recall_ma': recall_ma,
                           'f-score_ma': f_score_ma, 'mean_absolute_error': mean_absolute})
    else:
        mse_total = mean_squared_error(y_test, y_pred)

        mse_less_1 = mean_squared_error(np.array([float(el[0]) for el in zip(y_test, y_test2) if int(el[1]) == 1]),
                                        np.array([float(el[0][0]) for el in zip(y_pred, y_test2) if int(el[1]) == 1]))
        mse_more_1 = mean_squared_error(np.array([float(el[0]) for el in zip(y_test, y_test2) if int(el[1]) == 2]),
                                        np.array([float(el[0][0]) for el in zip(y_pred, y_test2) if int(el[1]) == 2]))

        rho = pearsonr([float(i) for i in y_test], [el[0] for el in y_pred])[0]
        result = {'mse_total': mse_total, 'mse_more_1': mse_more_1, 'mse_less_1': mse_less_1, 'rho': rho,
                  'accuracy_001': calculate_accuracy_with_threshold(y_test, y_pred, threshold=0.10),
                  'accuracy_025': calculate_accuracy_with_threshold(y_test, y_pred, threshold=0.25),
                  'accuracy_050': calculate_accuracy_with_threshold(y_test, y_pred, threshold=0.50),
                  'accuracy_100': calculate_accuracy_with_threshold(y_test, y_pred, threshold=1.00)}

    return result


def print_results(args, metrics):
    if args['dataset'] != Dataset.flood_heights:
        print("Accuracy ----------------> {}".format(metrics['accuracy']))

        if args['is_binary']:
            print("F-Score -----------------> {}".format(metrics['f-score']))
            print("Precision ---------------> {}".format(metrics['precision']))
            print("Recall ------------------> {}".format(metrics['recall']))

        else:
            print("Mean Absolute Error -----> {}".format(metrics['mean_absolute_error']))
            print("F-Score (macro) ---------> {}".format(metrics['f-score_ma']))
            print("Precision (macro) -------> {}".format(metrics['precision_ma']))
            print("Recall (macro) ----------> {}".format(metrics['recall_ma']))

    else:
        print("Mean Squared Error Total --> {}".format(metrics['mse_total']))
        print("Mean Squared Error < 1 m --> {}".format(metrics['mse_less_1']))
        print("Mean Squared Error > 1 m --> {}".format(metrics['mse_more_1']))
        print("Pearson Coefficient -------> {}".format(metrics['rho']))
        print("Accuracy 0.10 threshold ---> {}".format(metrics['accuracy_001']))
        print("Accuracy 0.25 threshold ---> {}".format(metrics['accuracy_025']))
        print("Accuracy 0.50 threshold ---> {}".format(metrics['accuracy_050']))
        print("Accuracy 1.00 threshold ---> {}".format(metrics['accuracy_100']))


def print_fold_results(args, metrics):
    if args['is_binary']:
        print("Accuracy ----------------> {}".format(metrics['accuracy']))
        print("F-Score -----------------> {}".format(metrics['f-score']))
    else:
        print("Accuracy ----------------> {}".format(metrics['accuracy']))
        print("F-Score (macro) ---------> {}".format(metrics['f-score_ma']))


def calculate_average_precision_ks(args, lst, ks=(50, 100, 250, 480)):
    if args['is_binary'] and args['dataset'] != Dataset.flood_heights:
        results, score = sorted(lst, key=itemgetter(0), reverse=True), 0

        for k in ks:
            y_score, y_true = zip(*results[:k])
            y_score, y_true = np.asarray(y_score), np.asarray(y_true)
            average_precision = average_precision_score(y_true.astype(int), y_score)
            score += average_precision
            print("Average Precision @ {:<3} -> {}".format(k, average_precision))

        print("Average Precision @ {} -> {}".format(ks, score / len(ks)))


def train_test_dense_net_split(args):
    filepath = check_path("weights/{}_split/weights.hdf5".format(args['dataset']))

    train_df = get_train_dataset_info(args['dataset'])
    test_df = get_test_dataset_info(args['dataset'])

    trn_flow, val_flow = get_training_and_validation_flow(args, train_df, split_size=0.10)
    tst_flow = create_flow(args, test_df, batch_size=1, shuffle=False)

    model = train_or_load_model(args, trn_flow, val_flow, filepath, trn_flow.n, val_flow.n)
    y_pred_prob = model.predict_generator(generator=tst_flow, verbose=1, steps=tst_flow.n)

    y_pred_prob, y_pred, y_test = calculate_prediction(args, y_pred_prob, trn_flow, test_df)
    metrics_dict = accuracy_precision_recall_fscore(args, y_test, y_pred)
    y_pred_prob_classes = list(zip(y_pred_prob.tolist(), y_test))

    print_classifications(args, tst_flow.filenames, test_df, y_pred)
    print_results(args, metrics_dict)

    calculate_average_precision_ks(args, y_pred_prob_classes)
    calculate_accuracy_per_class(args, y_test, y_pred)

    draw_class_activation_map(args, model, test_df)


def train_test_dense_net_split_regression(args):
    filepath = check_path("weights/{}_split/weights.hdf5".format(args['dataset']))

    train_df = get_train_dataset_info(args['dataset'])
    test_df = get_test_dataset_info(args['dataset'])

    trn_df_1 = pd.DataFrame(data={'filename': train_df['filename'].values, 'class': train_df['class'].values})
    trn_df_2 = pd.DataFrame(data={'filename': train_df['filename'].values, 'class': train_df['height'].values})

    test_data_frame_1 = pd.DataFrame(data={'filename': test_df['filename'].values, 'class': test_df['class'].values})
    test_data_frame_2 = pd.DataFrame(data={'filename': test_df['filename'].values, 'class': test_df['height'].values})

    trn_flow_1, val_flow_1 = get_training_and_validation_flow(args, trn_df_1, class_mode="categorical", split_size=0.10)
    trn_flow_2, val_flow_2 = get_training_and_validation_flow(args, trn_df_2, split_size=0.10)

    tst_flow_1 = create_flow(args, test_data_frame_1, batch_size=1, shuffle=False)
    tst_flow_2 = create_flow(args, test_data_frame_2, batch_size=1, shuffle=False)

    trn_flow = merge_output_generators(trn_flow_1, trn_flow_2)
    val_flow = merge_output_generators(val_flow_1, val_flow_2)
    tst_flow = merge_output_generators(tst_flow_1, tst_flow_2)

    model = train_or_load_model(args, trn_flow, val_flow, filepath, trn_flow_1.n, val_flow_1.n, branch="global")
    y_pred_prob = model.predict_generator(generator=tst_flow, verbose=1, steps=tst_flow_1.n)[1]
    y_pred_prob, y_pred, y_test = calculate_prediction(args, y_pred_prob, trn_flow, test_data_frame_2)

    metrics_dict = accuracy_precision_recall_fscore(args, y_test, y_pred, y_test2=test_data_frame_1['class'].values)
    print_classifications(args, tst_flow_1.filenames, test_data_frame_2, y_pred)
    print_results(args, metrics_dict)


def train_test_attention_guided_cnn_split_regression(args):
    train_df = get_train_dataset_info(args['dataset'])
    test_df = get_test_dataset_info(args['dataset'])

    first_branch_path = "weights/{}_attention_guided_global_branch/weights.hdf5"
    first_branch_path = check_path(first_branch_path.format(args['dataset']))

    second_branch_path = "weights/{}_attention_guided_local_branch/weights.hdf5"
    second_branch_path = check_path(second_branch_path.format(args['dataset']))

    all_network_path = "weights/{}_attention_guided_all/weights.hdf5"
    all_network_path = check_path(all_network_path.format(args['dataset']))

    trn_df_1_global = pd.DataFrame(data={'filename': train_df['filename'].values, 'class': train_df['class'].values})
    trn_df_2_global = pd.DataFrame(data={'filename': train_df['filename'].values, 'class': train_df['height'].values})

    tst_df_1_global = pd.DataFrame(data={'filename': test_df['filename'].values, 'class': test_df['class'].values})
    tst_df_2_global = pd.DataFrame(data={'filename': test_df['filename'].values, 'class': test_df['height'].values})

    trn_flow_1_global, val_flow_1_global = get_training_and_validation_flow(args, trn_df_1_global,
                                                                            class_mode="categorical", split_size=0.10)
    trn_flow_2_global, val_flow_2_global = get_training_and_validation_flow(args, trn_df_2_global, split_size=0.10)

    tst_flow_1_global = create_flow(args, tst_df_1_global, batch_size=1, shuffle=False)
    tst_flow_2_global = create_flow(args, tst_df_2_global, batch_size=1, shuffle=False)

    trn_flow_global = merge_output_generators(trn_flow_1_global, trn_flow_2_global)
    val_flow_global = merge_output_generators(val_flow_1_global, val_flow_2_global)
    tst_flow_global = merge_output_generators(tst_flow_1_global, tst_flow_2_global)

    model_global = train_or_load_model(args, trn_flow_global, val_flow_global, first_branch_path,
                                       trn_flow_1_global.n, val_flow_1_global.n, branch="global")
    y_pred_prob = model_global.predict_generator(generator=tst_flow_global, verbose=1, steps=tst_flow_1_global.n)[1]
    y_pred_prob, y_pred, y_test = calculate_prediction(args, y_pred_prob, trn_flow_global, tst_df_2_global)
    metrics_dict = accuracy_precision_recall_fscore(args, y_test, y_pred, y_test2=tst_df_1_global['class'].values)

    print("Global branch results.")
    print_results(args, metrics_dict)
    calculate_accuracy_per_class(args, y_test, y_pred)

    tst_local_df = crop_and_draw_class_activation_map(args, model_global, test_df)
    trn_local_df = crop_and_draw_class_activation_map(args, model_global, train_df)

    tst_local_df_1 = pd.DataFrame(data={'filename': tst_local_df['filename'].values,
                                        'class': tst_local_df['class'].values})
    tst_local_df_2 = pd.DataFrame(data={'filename': tst_local_df['filename'].values,
                                        'class': tst_local_df['height'].values})

    trn_local_df_1 = pd.DataFrame(data={'filename': trn_local_df['filename'].values,
                                        'class': trn_local_df['class'].values})
    trn_local_df_2 = pd.DataFrame(data={'filename': trn_local_df['filename'].values,
                                        'class': trn_local_df['height'].values})

    trn_flow_1_local, val_flow_1_local = get_training_and_validation_flow(args, trn_local_df_1,
                                                                          class_mode="categorical", split_size=0.10)
    trn_flow_2_local, val_flow_2_local = get_training_and_validation_flow(args, trn_local_df_2, split_size=0.10)

    tst_flow_1_local = create_flow(args, tst_local_df_1, batch_size=1, shuffle=False)
    tst_flow_2_local = create_flow(args, tst_local_df_2, batch_size=1, shuffle=False)

    trn_flow_local = merge_output_generators(trn_flow_1_local, trn_flow_2_local)
    val_flow_local = merge_output_generators(val_flow_1_local, val_flow_2_local)
    tst_flow_local = merge_output_generators(tst_flow_1_local, tst_flow_2_local)

    model_local = train_or_load_model(args, trn_flow_local, val_flow_local, second_branch_path,
                                      trn_flow_2_local.n, val_flow_2_local.n, branch="local")
    y_pred_prob = model_local.predict_generator(generator=tst_flow_local, verbose=1, steps=tst_flow_2_local.n)[1]
    y_pred_prob, y_pred, y_test = calculate_prediction(args, y_pred_prob, trn_flow_2_local, tst_local_df_2)
    metrics_dict = accuracy_precision_recall_fscore(args, y_test, y_pred, y_test2=tst_local_df_1['class'].values)

    print("Local branch results.")
    print_results(args, metrics_dict)
    calculate_accuracy_per_class(args, y_test, y_pred)

    trn_flow_merged = merge_all_generators(trn_flow_1_global, trn_flow_2_local)
    val_flow_merged = merge_all_generators(val_flow_1_global, val_flow_2_local)
    tst_flow_merged = merge_all_generators(tst_flow_1_global, tst_flow_2_local)

    models = {"model_global": model_global, "model_local": model_local}
    model = train_or_load_model(args, trn_flow_merged, val_flow_merged, all_network_path,
                                trn_flow_1_local.n, val_flow_1_local.n, branches_models=models)

    y_pred_prob = model.predict_generator(generator=tst_flow_merged, verbose=1, steps=tst_flow_1_local.n)[1]
    y_pred_prob, y_pred, y_test = calculate_prediction(args, y_pred_prob, trn_flow_1_global, tst_df_2_global)
    metrics_dict = accuracy_precision_recall_fscore(args, y_test, y_pred, y_test2=tst_df_1_global['class'].values)

    print("Fused model results.")
    print_results(args, metrics_dict)
    print_classifications(args, tst_flow_1_global.filenames, test_df, y_pred)
    calculate_accuracy_per_class(args, y_test, y_pred)


def train_test_attention_guided_cnn_split(args):
    train_df = get_train_dataset_info(args['dataset'])
    test_df = get_test_dataset_info(args['dataset'])

    first_branch_path = "weights/{}_attention_guided_global_branch/weights.hdf5"
    first_branch_path = check_path(first_branch_path.format(args['dataset']))

    second_branch_path = "weights/{}_attention_guided_local_branch/weights.hdf5"
    second_branch_path = check_path(second_branch_path.format(args['dataset']))

    all_network_path = "weights/{}_attention_guided_all/weights.hdf5"
    all_network_path = check_path(all_network_path.format(args['dataset']))

    trn_flow_1, val_flow_1 = get_training_and_validation_flow(args, train_df)
    model_global = train_or_load_model(args, trn_flow_1, val_flow_1, first_branch_path, trn_flow_1.n, val_flow_1.n)

    tst_flow_1 = create_flow(args, test_df, batch_size=1, shuffle=False)
    y_pred_prob = model_global.predict_generator(generator=tst_flow_1, verbose=1, steps=tst_flow_1.n)

    y_pred_prob, y_pred, y_test = calculate_prediction(args, y_pred_prob, trn_flow_1, test_df)
    metrics_dict = accuracy_precision_recall_fscore(args, y_test, y_pred)
    y_pred_prob_classes = list(zip(y_pred_prob.tolist(), y_test))

    print("Global branch results.")
    print_results(args, metrics_dict)
    calculate_accuracy_per_class(args, y_test, y_pred)
    calculate_average_precision_ks(args, y_pred_prob_classes)

    test_data_frame_2 = crop_and_draw_class_activation_map(args, model_global, test_df)
    train_data_frame_2 = crop_and_draw_class_activation_map(args, model_global, train_df)

    trn_flow_2, val_flow_2 = get_training_and_validation_flow(args, train_data_frame_2)
    model_local = train_or_load_model(args, trn_flow_2, val_flow_2, second_branch_path, trn_flow_2.n, val_flow_2.n)

    tst_flow_2 = create_flow(args, test_data_frame_2, batch_size=1, shuffle=False)
    y_pred_prob = model_local.predict_generator(generator=tst_flow_2, verbose=1, steps=tst_flow_2.n)
    y_pred_prob, y_pred, y_test = calculate_prediction(args, y_pred_prob, trn_flow_2, test_data_frame_2)
    metrics_dict = accuracy_precision_recall_fscore(args, y_test, y_pred)
    y_pred_prob_classes = list(zip(y_pred_prob.tolist(), y_test))

    print("Local branch results.")
    print_results(args, metrics_dict)
    calculate_accuracy_per_class(args, y_test, y_pred)
    calculate_average_precision_ks(args, y_pred_prob_classes)

    trn_flow_merged = merge_input_generators(trn_flow_1, trn_flow_2)
    val_flow_merged = merge_input_generators(val_flow_1, val_flow_2)
    tst_flow_merged = merge_input_generators(tst_flow_1, tst_flow_2)

    models = {"model_global": model_global, "model_local": model_local}
    model = train_or_load_model(args, trn_flow_merged, val_flow_merged, all_network_path,
                                trn_flow_1.n, val_flow_1.n, branches_models=models)

    y_pred_prob = model.predict_generator(generator=tst_flow_merged, verbose=1, steps=tst_flow_1.n)
    y_pred_prob, y_pred, y_test = calculate_prediction(args, y_pred_prob, trn_flow_1, test_df)
    metrics_dict = accuracy_precision_recall_fscore(args, y_test, y_pred)
    y_pred_prob_classes = list(zip(y_pred_prob.tolist(), y_test))

    print("Fused model results.")
    print_results(args, metrics_dict)
    print_classifications(args, tst_flow_1.filenames, test_df, y_pred)
    calculate_accuracy_per_class(args, y_test, y_pred)
    calculate_average_precision_ks(args, y_pred_prob_classes)


def train_test_attention_guided_cnn_cv(args):
    info = get_train_dataset_info(args['dataset'])
    x, y, = info.iloc[:, 0].values, info.iloc[:, 1].values

    metrics_dict_glob, y_pred_prob_classes_glob = defaultdict(int), []
    metrics_dict, y_pred_prob_classes, fl_nr = defaultdict(int), [], 1

    k_fold = StratifiedKFold(n_splits=args['nr_folds'], shuffle=True, random_state=args['random_seed'])
    # y_test_final, y_prob_final, y_pred_final = [], [], []

    for train, test in k_fold.split(x, y):
        train_data_frame = pd.DataFrame(data={'filename': x[train], 'class': y[train]})
        test_data_frame = pd.DataFrame(data={'filename': x[test], 'class': y[test]})

        first_branch_path = "weights/{}_attention_guided_global_branch_cv/weights_fold_{}_from_{}.hdf5"
        first_branch_path = check_path(first_branch_path.format(args['dataset'], fl_nr, args['nr_folds']))

        second_branch_path = "weights/{}_attention_guided_local_branch_cv/weights_fold_{}_from_{}.hdf5"
        second_branch_path = check_path(second_branch_path.format(args['dataset'], fl_nr, args['nr_folds']))

        all_network_path = "weights/{}_attention_guided_all_cv/weights_fold_{}_from_{}.hdf5"
        all_network_path = check_path(all_network_path.format(args['dataset'], fl_nr, args['nr_folds']))

        trn_flow_1, val_flow_1 = get_training_and_validation_flow(args, train_data_frame)
        model_global = train_or_load_model(args, trn_flow_1, val_flow_1, first_branch_path, trn_flow_1.n, val_flow_1.n)

        tst_flow_1 = create_flow(args, test_data_frame, batch_size=1, shuffle=False)
        y_pred_prob = model_global.predict_generator(generator=tst_flow_1, verbose=1, steps=tst_flow_1.n)
        y_pred_prob, y_pred, y_test = calculate_prediction(args, y_pred_prob, trn_flow_1, test_data_frame)
        metrics_it_glob = accuracy_precision_recall_fscore(args, y_test, y_pred)
        y_pred_prob_classes_glob += list(zip(y_pred_prob.tolist(), y_test))

        print("Global branch results.")
        print_fold_results(args, metrics_it_glob)
        calculate_accuracy_per_class(args, y_test, y_pred)

        test_data_frame_2 = crop_and_draw_class_activation_map(args, model_global, test_data_frame, fl_nr)
        train_data_frame_2 = crop_and_draw_class_activation_map(args, model_global, train_data_frame, fl_nr)

        trn_flow_2, val_flow_2 = get_training_and_validation_flow(args, train_data_frame_2)
        model_local = train_or_load_model(args, trn_flow_2, val_flow_2, second_branch_path, trn_flow_2.n, val_flow_2.n)

        tst_flow_2 = create_flow(args, test_data_frame_2, batch_size=1, shuffle=False)
        y_pred_prob = model_local.predict_generator(generator=tst_flow_2, verbose=1, steps=tst_flow_2.n)
        y_pred_prob, y_pred, y_test = calculate_prediction(args, y_pred_prob, trn_flow_2, test_data_frame_2)
        metrics_it = accuracy_precision_recall_fscore(args, y_test, y_pred)

        print("Local branch results.")
        print_fold_results(args, metrics_it)
        calculate_accuracy_per_class(args, y_test, y_pred)

        trn_flow_merged = merge_input_generators(trn_flow_1, trn_flow_2)
        val_flow_merged = merge_input_generators(val_flow_1, val_flow_2)
        tst_flow_merged = merge_input_generators(tst_flow_1, tst_flow_2)

        models = {"model_global": model_global, "model_local": model_local}
        model = train_or_load_model(args, trn_flow_merged, val_flow_merged, all_network_path,
                                    trn_flow_1.n, val_flow_1.n, branches_models=models)

        y_pred_prob = model.predict_generator(generator=tst_flow_merged, verbose=1, steps=tst_flow_1.n)
        y_pred_prob, y_pred, y_test = calculate_prediction(args, y_pred_prob, trn_flow_1, test_data_frame)
        metrics_it = accuracy_precision_recall_fscore(args, y_test, y_pred)
        y_pred_prob_classes += list(zip(y_pred_prob.tolist(), y_test))

        print("Fused model results.")
        print_fold_results(args, metrics_it)
        print_classifications(args, tst_flow_1.filenames, test_data_frame, y_pred)
        calculate_accuracy_per_class(args, y_test, y_pred)

        draw_class_activation_map(args, model, test_data_frame)

        # y_pred_final += y_pred
        # y_test_final += y_test.astype(int).tolist()
        # y_prob_final.extend([x.tolist() for x in y_pred_prob])

        metrics_dict = dict((k, metrics_dict[k] + v) for k, v in metrics_it.items())
        metrics_dict_glob = dict((k, metrics_dict_glob[k] + v) for k, v in metrics_it_glob.items())
        K.clear_session()
        fl_nr += 1

    print("Global branch results.")
    metrics_dict_glob = dict((k, v / args['nr_folds']) for k, v in metrics_dict_glob.items())
    print_results(args, metrics_dict_glob)
    calculate_average_precision_ks(args, y_pred_prob_classes_glob)

    # precision, recall, average_precision = dict(), dict(), dict()
    # y_test_bin_final = label_binarize(y_test_final, classes=[0, 1, 2])
    # y_prob_final = np.array(y_prob_final)
    # for i in range(3):
    #     precision[i], recall[i], _ = precision_recall_curve(y_test_bin_final[:, i],
    #                                                         y_prob_final[:, i])
    #     average_precision[i] = average_precision_score(y_test_bin_final[:, i], y_prob_final[:, i])
    #
    # colors = cycle(['navy', 'turquoise', 'darkorange'])
    # tags = ["no flood", "water < 1m", "water > 1m"]
    # lines = []
    # labels = []
    # for i, color in zip(range(3), colors):
    #     l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    #     lines.append(l)
    #     labels.append('Precision-recall for {0} (area = {1:0.2f})'
    #                   ''.format(tags[i], average_precision[i]))
    #
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend(lines, labels)
    # plt.savefig('prec_rec.png')
    #
    # print(confusion_matrix(y_test_final, y_pred_final))

    print("Fused model results.")
    metrics_dict = dict((k, v / args['nr_folds']) for k, v in metrics_dict.items())
    print_results(args, metrics_dict)
    calculate_average_precision_ks(args, y_pred_prob_classes)


def train_test_dense_net_cv(args):
    info = get_train_dataset_info(args['dataset'])
    x, y, = info.iloc[:, 0].values, info.iloc[:, 1].values

    metrics_dict, y_pred_prob_classes, fold_nr = defaultdict(int), [], 1
    k_fold = StratifiedKFold(n_splits=args['nr_folds'], shuffle=True, random_state=args['random_seed'])

    for train, test in k_fold.split(x, y):
        filepath = "weights/{}_cv/weights_fold_{}_from_{}.hdf5"
        filepath = check_path(filepath.format(args['dataset'], fold_nr, args['nr_folds']))

        train_data_frame = pd.DataFrame(data={'filename': x[train], 'class': y[train]})
        test_data_frame = pd.DataFrame(data={'filename': x[test], 'class': y[test]})

        trn_flow, val_flow = get_training_and_validation_flow(args, train_data_frame)
        tst_flow = create_flow(args, test_data_frame, batch_size=1, shuffle=False)

        model = train_or_load_model(args, trn_flow, val_flow, filepath, trn_flow.n, val_flow.n)
        y_pred_prob = model.predict_generator(generator=tst_flow, verbose=1, steps=tst_flow.n)

        y_pred_prob, y_pred, y_test = calculate_prediction(args, y_pred_prob, trn_flow, test_data_frame)
        metrics_it = accuracy_precision_recall_fscore(args, y_test, y_pred)
        y_pred_prob_classes += list(zip(y_pred_prob.tolist(), y_test))

        print_fold_results(args, metrics_it)
        print_classifications(args, tst_flow.filenames, test_data_frame, y_pred)
        calculate_accuracy_per_class(args, y_test, y_pred)
        draw_class_activation_map(args, model, test_data_frame)

        metrics_dict = dict((k, metrics_dict[k] + v) for k, v in metrics_it.items())
        K.clear_session()
        fold_nr += 1

    metrics_dict = dict((k, v / args['nr_folds']) for k, v in metrics_dict.items())
    print_results(args, metrics_dict)
    calculate_average_precision_ks(args, y_pred_prob_classes)


def print_classifications(args, filenames, tst_df, y_pred):
    if not args['print_classifications']:
        return

    with open("info.txt", "a+") as f:
        for idx, el in enumerate(y_pred):
            green, red, end = '\033[92m', '\033[91m', '\033[0m'
            color = green if tst_df['class'].values[idx] == el else red
            print(color + "{:<105} -> True: {} | Pred: {}"
                  .format(filenames[idx], tst_df['class'].values[idx], el) + end)
            tick = "Correct" if tst_df['class'].values[idx] == el else "Incorrect"
            f.write("{:<105} -> True: {} | Pred: {} -> {}\n"
                    .format(filenames[idx], tst_df['class'].values[idx], el, tick))
        f.write("------------------------------------------------\n")
