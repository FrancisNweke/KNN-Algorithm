from math import sqrt
import pandas as pd


# Step 1 - calculate the distance btw two rows in a dataset using euclidean or manhattan
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


def manhattan_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += abs(row1[i] - row2[i])
    return distance


# Step 2 - get the nearest neighbors
# Locate the most similar neighbors - using euclidean distance
def get_neighbors_eud(train_dataset, test_row, num_neighbors):
    distances = list()
    for train_row in train_dataset:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# using manhattan distance
def get_neighbors_man(train_dataset, test_row, num_neighbors):
    distances = list()
    for train_row in train_dataset:
        dist = manhattan_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Step 3 - predict classification
def predict_class(neighbors: list):
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# Other useful functions for data preprocessing
# Load data file and convert to list
def load_data_file(filename) -> list:
    data_file = pd.read_csv(f'{filename}.csv', skipinitialspace=True, skip_blank_lines=True)
    return data_file.values.tolist()


# Convert column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column])


# Convert class column to integer
def str_column_class_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()

    for i, value in enumerate(unique):
        lookup[value] = i
        print(f'{value} => {i}')
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the minimum and maximum values of each row
def find_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        min_value = min(col_values)
        max_value = max(col_values)
        minmax.append([min_value, max_value])
    return minmax


# rescale dataset columns to range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Tutorial Link: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

# Distance metrics:
# https://medium.com/@kunal_gohrani/different-types-of-distance-metrics-used-in-machine-learning-e9928c5e26c7
# https://www.analyticsvidhya.com/blog/2020/02/4-types-of-distance-metrics-in-machine-learning/
