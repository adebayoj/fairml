import matplotlib

# temporary work around down to virtualenv
# matplotlib issue.
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['figure.figsize'] = (8, 8)
plt.style.use('ggplot')


def plot_dependencies(dictionary_values,
                      pos_color="#3DE8F7",
                      negative_color="#ff4d4d",
                      reverse_values=False,
                      title=""):

    # add check to make sure that dependence features are not zeros
    if np.sum(np.array(dictionary_values.values())) == 0.0:
        print("Feature dependence for all attributes equal zero."
              " There is nothing to plot here. ")
        return None

    column_names = list(dictionary_values.keys())
    coefficient_values = list(dictionary_values.values())

    # get maximum
    maximum_value = np.absolute(np.array(coefficient_values)).max()
    coefficient_values = ((np.array(coefficient_values) / maximum_value) * 100
                          )

    index_sorted = np.argsort(np.array(coefficient_values))
    sorted_column_names = list(np.array(column_names)[index_sorted])
    sorted_column_values = list(np.array(coefficient_values)[index_sorted])

    pos = np.arange(len(sorted_column_values)) + 0.7

    # rearrange this at some other point.
    def assign_colors_to_bars(array_values,
                              pos_influence=pos_color,
                              negative_influence=negative_color,
                              reverse=reverse_values):

        # if you want the colors to be reversed for positive
        # and negative influences.
        if reverse:
            pos_influence, negative_influence = (negative_influence,
                                                 pos_influence)

        # could rewrite this as a lambda function
        # but I understand this better
        def map_x(x):
            if x > 0:
                return pos_influence
            else:
                return negative_influence
        bar_colors = list(map(map_x, array_values))
        return bar_colors

    fig = plt.figure(1)

    bar_colors = assign_colors_to_bars(coefficient_values, reverse=True)
    bar_colors = list(np.array(bar_colors)[index_sorted])

    plt.barh(pos, sorted_column_values, align='center', color=bar_colors)
    plt.yticks(pos, sorted_column_names)
    plt.xlim(-105, 105)

    if title:
        plt.title("{}".format(title))

    return fig
