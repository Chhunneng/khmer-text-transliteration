import matplotlib.pyplot as plt
from predict import predict_khmer


def calculate_accuracy(dataset, test_size=0.2):
    """
    Calculate the accuracy of the Khmer text transliteration model.

    This function evaluates the accuracy of the transliteration model by comparing
    the predicted Khmer text with the true Khmer text for a subset of the dataset.

    Args:
        dataset (list of tuples): A list of tuples where each tuple contains an English text
                                  and its corresponding true Khmer text.
        test_size (float, optional): The proportion of the dataset to be used for testing.
                                     Defaults to 0.2 (20%).

    Returns:
        float: The accuracy of the model, calculated as the number of correct predictions
               divided by the total number of test samples.
    """
    test_samples = int(len(dataset) * test_size)
    correct = 0
    for eng, true_khm in dataset[:test_samples]:
        pred_khm = predict_khmer(eng)
        if pred_khm == true_khm:
            correct += 1
    return correct / test_samples


def plot_history(history):
    """
    Plots the training and validation loss over epochs.

    Args:
        history (History): A History object returned by the fit method of a Keras model.
                           It contains the training and validation loss values for each epoch.

    Returns:
        None
    """
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training Progress")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
