import numpy as np
import pandas as pd
import random

from utils import get_type_of_feature, calculate_overall_metric, calculate_mse, calculate_entropy


# Verify the data is pure, whether all the data points belong to a single class and predicted value is same.
def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


def create_leaf(data, ml_task):
    label_column = data[:, -1]
    if ml_task == "regression":
        leaf = np.mean(label_column)
    elif ml_task == "classification":
        unique_classes, counts_unique_classes = np.unique(label_column,
                                                          return_counts=True)
        index = counts_unique_classes.argmax()
        leaf = unique_classes[index]
    else:
        raise Exception("Unknown ML task :", ml_task)
    return leaf


# Potential split could be to split at all the unique values of the data
def get_potential_splits(data, random_subspace):
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns - 1))

    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(population=column_indices,
                                       k=random_subspace)

    for column_index in column_indices:  # excluding the last column which is the response
        values = data[:, column_index]
        unique_values = np.unique(values)

        potential_splits[column_index] = unique_values

    return potential_splits


def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]

    # feature is categorical
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]

    return data_below, data_above


def determine_best_split(data, potential_splits, ml_task):
    first_iteration = True
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data,
                                                split_column=column_index,
                                                split_value=value)

            if ml_task == "regression":
                current_overall_metric = calculate_overall_metric(data_below,
                                                                  data_above,
                                                                  metric_function=calculate_mse)

            elif ml_task == "classification":
                current_overall_metric = calculate_overall_metric(data_below,
                                                                  data_above,
                                                                  metric_function=calculate_entropy)
            else:
                raise Exception("Unknown ML task :", ml_task)

            if first_iteration or current_overall_metric <= best_overall_metric:
                first_iteration = False

                best_overall_metric = current_overall_metric
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


def decision_tree_algorithm(df, ml_task, counter=0, min_samples=2, max_depth=5, random_subspace=None):
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = get_type_of_feature(df)
        data = df.values
    else:
        data = df

    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        leaf = create_leaf(data, ml_task)
        return leaf
    else:
        counter += 1
        potential_splits = get_potential_splits(data, random_subspace)
        split_column, split_value = determine_best_split(data, potential_splits, ml_task)
        data_below, data_above = split_data(data, split_column, split_value)

        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            leaf = create_leaf(data, ml_task)
            return leaf

        #  Decision
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)

        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)

        # instantiate sub-tree
        sub_tree = {question: []}

        yes_answer = decision_tree_algorithm(data_below,
                                             ml_task,
                                             counter,
                                             min_samples,
                                             max_depth,
                                             random_subspace)
        no_answer = decision_tree_algorithm(data_above,
                                            ml_task,
                                            counter,
                                            min_samples,
                                            max_depth,
                                            random_subspace)
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree