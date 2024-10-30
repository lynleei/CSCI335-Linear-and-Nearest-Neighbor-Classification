################################################################
# Machine Learning
# Assignment 1 Starting Code
#
# Author: R. Zanibbi
# Author: E. Lima
# Author: Lynlee Hong (RIT Student) lh3843@g.rit.edu
################################################################


import argparse
import numpy as np
# import statistics as stats
import matplotlib.pyplot as plt

################################################################
# Metrics and visualization
################################################################


def conf_matrix(class_matrix, data_matrix, title=None, print_results=False):

    # Initialize confusion matrix with 0's
    confusions = np.zeros((2, 2), dtype=int)

    # Generate output/target pairs, convert to list of integer pairs
    # * '-1' indexing indicates the final column
    # * list(zip( ... )) combines the output and target class label columns into a list of pairs
    out_target_pairs = [
        (int(out), int(target))
        for (out, target) in list(zip(class_matrix[:, -1], data_matrix[:, -1]))
    ]

    # Use output/target pairs to compile confusion matrix counts
    for (out, target) in out_target_pairs:
        confusions[out][target] += 1

    # Compute recognition rate
    inputs_correct = confusions[0][0] + confusions[1][1]
    inputs_total = np.sum(confusions)
    recognition_rate = inputs_correct / inputs_total * 100

    if print_results:
        if title:
            print("\n>>>  " + title)
        print(
            "\n    Recognition rate (correct / inputs):\n    ", recognition_rate, "%\n"
        )
        print("    Confusion Matrix:")
        print("              0: Blue-True  1: Org-True")
        print("---------------------------------------")
        print("0: Blue-Pred |{0:12d} {1:12d}".format(confusions[0][0], confusions[0][1]))
        print("1: Org-Pred  |{0:12d} {1:12d}".format(confusions[1][0], confusions[1][1]))

    return (recognition_rate, confusions)


def draw_results(data_matrix, class_fn, title, file_name):

    # Fix axes ranges so that X and Y directions are identical (avoids 'stretching' in one direction or the other)
    # Use numpy amin function on first two columns of the training data matrix to identify range
    pad = 0.25
    min_tick = np.amin(data_matrix[:, 0:2]) - pad
    max_tick = np.amax(data_matrix[:, 0:2]) + pad
    plt.xlim(min_tick, max_tick)
    plt.ylim(min_tick, max_tick)

    ##################################
    # Grid dots to show class regions
    ##################################

    axis_tick_count = 125 # increased for more clear view of plot
    x = np.linspace(min_tick, max_tick, axis_tick_count, endpoint=True)
    y = np.linspace(min_tick, max_tick, axis_tick_count, endpoint=True)
    (xx, yy) = np.meshgrid(x, y)
    grid_points = np.concatenate(
        (xx.reshape(xx.size, 1), yy.reshape(yy.size, 1)), axis=1
    )

    class_out = class_fn(grid_points)

    # Separate rows for blue (0) and orange (1) outputs, plot separately with color
    blue_points = grid_points[np.where(class_out[:, 1] < 1.0)]
    orange_points = grid_points[np.where(class_out[:, 1] > 0.0)]

    plt.scatter(
        blue_points[:, 0],
        blue_points[:, 1],
        marker=".",
        s=1,
        facecolors="blue",
        edgecolors="blue",
        alpha=0.4,
    )
    plt.scatter(
        orange_points[:, 0],
        orange_points[:, 1],
        marker=".",
        s=1,
        facecolors="orange",
        edgecolors="orange",
        alpha=0.4,
    )

    ##################################
    # Decision boundary (black line)
    ##################################

    # ONE method for ALL classifier types

    # reshape the class predictions to match the grid shape
    class_grid = class_out[:, 1].reshape(xx.shape)

    # compute differences along x and y axes
    diff_x = np.diff(class_grid, axis=1)
    diff_y = np.diff(class_grid, axis=0)

    # where the class changes happens
    boundary_x = np.argwhere(diff_x != 0)
    boundary_y = np.argwhere(diff_y != 0)

    delta = (max_tick - min_tick) / axis_tick_count / 2  # half length of line segments

    # plot vertical boundaries
    for (i, j) in boundary_x:
        x_boundary = (xx[i, j + 1] + xx[i, j]) / 2
        y_boundary = yy[i, j]
        plt.plot(
            [x_boundary, x_boundary],
            [y_boundary - delta, y_boundary + delta],
            color='black', linewidth=0.5
        )

    # plot horizontal boundaries
    for (i, j) in boundary_y:
        x_boundary = xx[i, j]
        y_boundary = (yy[i + 1, j] + yy[i, j]) / 2
        plt.plot(
            [x_boundary - delta, x_boundary + delta],
            [y_boundary, y_boundary],
            color='black', linewidth=0.5
        )

    ##################################
    # Show training samples
    ##################################

    # Separate rows for blue (0) and orange (1) target inputs, plot separately with color
    blue_targets = data_matrix[np.where(data_matrix[:, 2] < 1.0)]
    orange_targets = data_matrix[np.where(data_matrix[:, 2] > 0.0)]

    plt.scatter(
        blue_targets[:, 0],
        blue_targets[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="blue",
    )
    plt.scatter(
        orange_targets[:, 0],
        orange_targets[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="darkorange",
    )
    ##################################
    # Add title and write file
    ##################################

    # Set title and save plot if file name is passed (extension determines type)
    plt.title(title)
    plt.savefig(file_name)
    print("\nWrote image file: " + file_name)
    plt.close()


################################################################
# Interactive Testing
################################################################
def test_points(data_matrix, beta_hat):

    print("\n>> Interactive testing for (x_1,x_2) inputs")
    stop = False
    while True:
        x = input("\nEnter x_1 ('stop' to end): ")

        if x == "stop":
            break
        else:
            x = float(x)

        y = float(input("Enter x_2: "))
        k = int(input("Enter k: "))

        lc = linear_classifier(beta_hat)
        knn = knn_classifier(k, data_matrix)

        print("   least squares: " + str(lc(np.array([x, y]).reshape(1, 2))))
        print("             knn: " + str(knn(np.array([x, y]).reshape(1, 2))))


################################################################
# Classifiers
################################################################

def linear_classifier(weight_vector):
    # Constructs a linear classifier
    def classifier(input_matrix):
        # Take the dot product of each sample ( data_matrix row ) with the weight_vector (col vector)
        # -- as defined in Hastie equation (2.2)

        # Check if expected number of columns for input_matrix
        if input_matrix.shape[1] == len(weight_vector) - 1:
            X = input_matrix  # Use input_matrix directly
        elif input_matrix.shape[1] == len(weight_vector):
            # If input_matrix has labels, exclude last col
            X = input_matrix[:, :-1]
        else:
            raise ValueError("input_matrix has incorrect number of columns")

        # adds bias
        X_bias = np.c_[np.ones(X.shape[0]), X]

        # compute scores
        scores = np.dot(X_bias, weight_vector)

        # compute predicted classes, threshold at 0.5
        predicted_class = (scores >= 0.5).astype(float)

        # Return N x 2 result array: [scores, predicted_class]
        return np.column_stack((scores, predicted_class))

    return classifier


def find_optimal_weights(input_matrix):
    """
    Function computes the least squares optimal weight vector using the method of least squares.
    - input_matrix: The full data matrix where the last column is the label.
    """
    X = input_matrix[:, :-1]  # all X except last
    y = input_matrix[:, -1]  # gets y (last col)

    # Add bias to X matrix
    X_bias = np.c_[np.ones(X.shape[0]), X]

    # Compute weight vector using least squares formula
    weights = np.linalg.pinv(X_bias.T @ X_bias) @ (X_bias.T @ y)

    return weights


def knn_classifier(k, data_matrix):
    # Constructs a knn classifier for the passed value of k and training data matrix
    def classifier(input_matrix):
        # Determine if input_matrix includes labels
        if input_matrix.shape[1] == data_matrix.shape[1]:
            # Assume input_matrix has labels, use all cols except last
            X_input = input_matrix[:, :-1]
        elif input_matrix.shape[1] == data_matrix.shape[1] - 1:
            X_input = input_matrix  # else use as is
        else:
            raise ValueError("input_matrix has incorrect number of columns")

        # Extract X and y from data_matrix
        X_train = data_matrix[:, :-1]
        y_train = data_matrix[:, -1]

        (input_count, _) = X_input.shape
        result = np.zeros((input_count, 2))  # N x 2 matrix to store predictions

        for i in range(input_count):
            # Calculate Euclidean distance between input point and all training points
            distances = np.sqrt(np.sum((X_train - X_input[i, :]) ** 2, axis=1))

            # Find indices of the k nearest neighbors
            k_indices = np.argsort(distances)[:k]

            # Get the labels of the k nearest neighbors
            k_nearest_labels = y_train[k_indices]

            # Determine the majority class
            classes, counts = np.unique(k_nearest_labels, return_counts=True)
            majority_class = classes[np.argmax(counts)]

            result[i, 1] = majority_class  # Store the predicted class in result array

        return result

    return classifier


################################################################
# Main function
################################################################


def main():
    # Process arguments using 'argparse'
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="numpy data matrix file containing samples")
    args = parser.parse_args()

    # Load data
    data_matrix = np.load(args.data_file)
    (confm, rr) = conf_matrix(
        data_matrix, data_matrix, "Data vs. Itself Sanity Check", print_results=True
    )

    optimal_weights = find_optimal_weights(data_matrix)

    # Construct linear classifier
    lc = linear_classifier(optimal_weights) # changed to use helper function
    lsc_out = lc(data_matrix)

    # Compute results on training set
    conf_matrix(lsc_out, data_matrix, "Least Squares", print_results=True)
    draw_results(data_matrix, lc, "Least Squares Linear Classifier", "ls.pdf")

    # Nearest neighbor
    for k in [1, 15]:
        knn = knn_classifier(k, data_matrix)
        knn_out = knn(data_matrix)
        conf_matrix(knn_out, data_matrix, "knn: k=" + str(k), print_results=True)
        draw_results(
            data_matrix,
            knn,
            "k-NN Classifier (k=" + str(k) + ")",
            "knn-" + str(k) + ".pdf",
        )

    # Interactive testing
    test_points(data_matrix, optimal_weights)


if __name__ == "__main__":
    main()
