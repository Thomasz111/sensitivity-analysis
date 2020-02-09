import numpy as np


def get_three_ranges(a, b, c, d, number_of_samples, full_range):
    # we define starting and ending points, so assuming they are 0 - 20 and sampling is 0.01 we have 2000 points
    # 0, 0.01, 0.02 ... etc
    first_x_range_start, first_x_range_stop = (a, b)
    second_x_range_start, second_x_range_stop = (b, c)
    third_x_range_start, third_x_range_stop = (c, d)

    # 0, 0.01, 0.02 .... 19.99
    first_x_data = full_range[first_x_range_start * number_of_samples: first_x_range_stop * number_of_samples]
    first_x_data = np.array(first_x_data).reshape((1, len(first_x_data)))

    # 20.00, 20.01, 20.02 .... 39.99
    second_x_data = full_range[second_x_range_start * number_of_samples: second_x_range_stop * number_of_samples]
    second_x_data = np.array(second_x_data).reshape((1, len(second_x_data)))

    # 40.00, 40.01, 40.02 .... 59.99
    third_x_data = full_range[third_x_range_start * number_of_samples:third_x_range_stop * number_of_samples]
    third_x_data = np.array(third_x_data).reshape((1, len(third_x_data)))

    return first_x_data, second_x_data, third_x_data
