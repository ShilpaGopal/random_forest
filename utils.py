import pandas as pd
import numpy as np
import random


# Split the data based on test size
def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices,
                                 k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


"""
Get the type of feature from data frame to determine the split operator
1. String can be considered as categorical
2. No of unique values is less than some threshold values then it can be considered as categorical
3. Any number of values greater than given threshold between some range can be considered as continuous values
"""


def get_type_of_feature(df):
    feature_types = []
    n_unique_values_threshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            sample_value = unique_values[0]

            if (isinstance(sample_value, str)) or (len(unique_values) <= n_unique_values_threshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")

    return feature_types


"""
To check the purity of the split 
Mean squared error is used for regression
Entropy is used for classification
"""


def calculate_mse(data):
    actual_values = data[:, -1]

    if len(actual_values) == 0:  # empty data
        mse = 0
    else:
        prediction = np.mean(actual_values)
        mse = np.mean((actual_values - prediction) ** 2)

    return mse


def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column,
                          return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def calculate_overall_metric(data_below, data_above, metric_function):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_metric = (p_data_below * metric_function(data_below)
                      + p_data_above * metric_function(data_above))

    return overall_metric


def predict_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # Decision
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    # recursive
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)


def make_predictions(df, tree):
    if len(df) != 0:
        predictions = df.apply(predict_example,
                               args=(tree,),
                               axis=1)
    else:
        predictions = pd.Series()
    return predictions


def calculate_accuracy(df, tree):
    df["classification"] = df.apply(predict_example,
                                    args=(tree,),
                                    axis=1)
    df["classification_correct"] = df["classification"] == df["label"]
    accuracy = df["classification_correct"].mean()
    return accuracy


def calculate_r_squared(df, tree):
    labels = df.label
    mean = labels.mean()
    predictions = df.apply(predict_example,
                           args=(tree,),
                           axis=1)

    ss_res = sum((labels - predictions) ** 2)
    ss_tot = sum((labels - mean) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return r_squared


def r_squared(predictions, labels):
    mean = labels.mean()
    ss_res = sum((labels - predictions) ** 2)
    ss_tot = sum((labels - mean) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return r_squared


def accuracy(predictions, labels):
    predictions_correct = predictions == labels
    accuracy = predictions_correct.mean()

    return accuracy
