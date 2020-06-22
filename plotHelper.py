import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


def plot_decision_boundaries(tree, x_min, x_max, y_min, y_max):
    color_keys = {True: "orange", False: "blue"}

    # recursive part
    if isinstance(tree, dict):
        question = list(tree.keys())[0]
        yes_answer, no_answer = tree[question]
        feature, _, value = question.split()

        if feature == "x":
            plot_decision_boundaries(yes_answer, x_min, float(value), y_min, y_max)
            plot_decision_boundaries(no_answer, float(value), x_max, y_min, y_max)
        else:
            plot_decision_boundaries(yes_answer, x_min, x_max, y_min, float(value))
            plot_decision_boundaries(no_answer, x_min, x_max, float(value), y_max)

    # "tree" is a leaf
    else:
        plt.fill_between(x=[x_min, x_max], y1=y_min, y2=y_max, alpha=0.2, color=color_keys[tree])

    return


def create_plot(df, tree=None, title=None):
    sns.lmplot(data=df,
               x="x",
               y="y",
               hue="label",
               fit_reg=False,
               height=4,
               aspect=1.5,
               legend=False)
    plt.title(title)

    if tree or tree == False:  # root of the tree might just be a leave with "False"
        x_min, x_max = round(df.x.min()), round(df.x.max())
        y_min, y_max = round(df.y.min()), round(df.y.max())

        plot_decision_boundaries(tree, x_min, x_max, y_min, y_max)
    return