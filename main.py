from models.knn_algorithm import euclidean_distance, manhattan_distance
from models.knn_algorithm import get_neighbors_eud, get_neighbors_man, predict_class
from models.knn_algorithm import load_data_file, str_column_to_float, str_column_class_to_int

# Test distance function
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]

row0 = dataset[0]

for row in dataset:
    distance = manhattan_distance(row0, row)
    print(distance)

num_neighbors = 4
neighbors = get_neighbors_man(dataset, dataset[4], num_neighbors)
print(f'\n{num_neighbors} nearest neighbors to {dataset[4]} are: ')
for neighbor in neighbors:
    print(neighbor)

prediction = predict_class(neighbors)

print(f'\nActual: {dataset[4][-1]} || Prediction: {prediction}\n')

print('\t\t\t\tKNN on the Iris dataset')
# make a prediction using KNN on Iris dataset
# load data from file
data_file = load_data_file('data/iris')
for column in range(len(data_file[0]) - 1):
    str_column_to_float(data_file, column)

# convert class column to integers
class_column = len(data_file[0])
str_column_class_to_int(data_file, class_column - 1)

row = [5.7, 3.8, 1.7, 0.3]

num_neighbors = 9

nearest_neighbors = get_neighbors_man(data_file, row, num_neighbors)

print('\n')  # Go to the next line
for row in nearest_neighbors:
    print(row)

label = predict_class(nearest_neighbors)

print(f'\nData row: {row} => {label}')
