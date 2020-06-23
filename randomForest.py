import numpy as np
import pandas as pd
from decisionTreeAlgorithm import decision_tree_algorithm
from utils import make_predictions


def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0,
                                          high=len(train_df),
                                          size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    return df_bootstrapped


def random_forest_algorithm(train_df, ml_task, n_trees, n_bootstrap, min_samples, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped,
                                       ml_task=ml_task,
                                       min_samples=min_samples,
                                       max_depth=dt_max_depth,
                                       random_subspace=n_features)
        forest.append(tree)

    return forest


def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = make_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_prediction = df_predictions.mean(axis=1)

    return random_forest_prediction


def random_forest_classification(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = make_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_prediction = df_predictions.mode(axis=1)[0]

    return random_forest_prediction
