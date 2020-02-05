"""
Program that uses CNNs supervised learning to classify IMDB movie reviews as positive or negative (sentiment/polarity detection)
Consulted and compared sources that solve the problem (with less accuracy than this program): [1] https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py, [2] https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/
"""

from __future__ import print_function
import numpy as np
from scipy import stats
from keras import utils
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sn


def get_data_splits(train_percent, valid_percent, vocab_size, top_common_words_to_skip, words_per_text):

    """Splits the data set into 3 partitions (training, validation and test) accordingly to the passed split fractions"""

    # load the full data set
    (x0, y0), (x1, y1) = imdb.load_data(num_words=vocab_size, skip_top=top_common_words_to_skip)
    x = np.concatenate((x0, x1))
    y = np.concatenate((y0, y1))

    # study the data
    describe_data(x, y)

    # pre-process the data
    x, y = pre_process_data(x, y, words_per_text)

    # split the data in the three sets
    full_len = len(x)
    train_len = round(train_percent * full_len)
    valid_len = round(valid_percent * full_len)
    test_len = full_len - train_len - valid_len
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_valid = x[train_len:-test_len]
    y_valid = y[train_len:-test_len]
    x_test = x[-test_len:]
    y_test = y[-test_len:]

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def describe_data(x, y):

    """Statistically describe the data"""

    # analyse polarity label distribution
    print("Data set statistical analysis:\n\tPolarity distribution:\n\t\tPositive:\t{}\n\t\tNegative:\t{}\n".format(len([label for label in y if label == 1]), len([label for label in y if label == 0])))

    # analyse the length of the texts
    text_lengths = np.array([len(text) for text in x])
    _, min_max, mean, var, skew, kurt = stats.describe(text_lengths)
    print("\tText length:\n\t\tMin:\t\t\t{}\n\t\tMax:\t\t\t{}\n\t\tMean:\t\t\t{}\n\t\tStd:\t\t\t{}\n\t\tSkewness:\t\t{}\n\t\tKurtosis:\t\t{}".format(min_max[0], min_max[1], round(mean, 3), round(np.math.sqrt(var), 3), round(skew, 3), round(kurt, 3)))
    print("\t\t90% of the texts have ", sorted(text_lengths)[round(len(text_lengths)*0.1)], "-",  sorted(text_lengths)[round(len(text_lengths)*0.9)], "words")
    print("\t\t95% of the texts have no more than", sorted(text_lengths)[round(len(text_lengths)*0.95)], "words")
    print("\t\t99.9% of the texts have no more than", sorted(text_lengths)[round(len(text_lengths)*0.999)], "words")
    plt.boxplot(text_lengths)
    plt.title("Text length boxplot")
    plt.ylabel("Text length")
    plt.savefig("imdb_cnn_text_length_boxplot.png")

    # analyse the total number of different words in the data set
    unique_words = set()
    for text in x:
        for word in text:
            unique_words.add(word)
    print("\tTotal number of words through all texts:", len(unique_words), "\n")


def pre_process_data(x, y, words_per_text):

    """Pre-process the data"""

    # shuffle the features and output labels (in parallel)
    np.random.seed(1)
    zipped_data = list(zip(x, y))
    np.random.shuffle(zipped_data)
    x, y = zip(*zipped_data)
    np.random.seed()

    # make all feature vectors in x have same number of words, truncating longer ones and filling shorter ones with a special value
    x = sequence.pad_sequences(x, maxlen=words_per_text)

    return x, np.array(y)


def build_model(embedding_dim, vocab_size, words_per_text, filter_num, mask_size, dropout_ratio, hidden_units, optimizer, loss_function):

    """Build and compile a neural network model"""

    # use a linear layer stack
    model = Sequential()

    # set up a layer of word embeddings created by mapping the word identifiers fo the texts to a high dimensional space
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=words_per_text))

    # use dropout to ignore some input units during training and reduce overfitting
    model.add(Dropout(dropout_ratio))

    # use 1D convolution in the text identifiers to learn association of words that appear together in the text
    model.add(Conv1D(filters=filter_num, kernel_size=mask_size, activation="relu"))

    # reduce dimensionality with a max-pooling layer
    model.add(GlobalMaxPooling1D())

    # increase the model complexity
    model.add(Dense(hidden_units, activation="relu"))

    # use dropout to ignore some input units during training and reduce overfitting
    model.add(Dropout(dropout_ratio))

    # the output layer has a single unit, and uses sigmoid activation function, suitable for binary classification
    model.add(Dense(1, activation="sigmoid"))

    # compile the model
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    return model


def train_model(model, x_train, y_train, x_valid, y_valid, batch_size, epoch_num):

    """Train the model"""

    return model.fit(x_train, y_train, validation_data=[x_valid, y_valid], batch_size=batch_size, epochs=epoch_num)


def evaluate_model(model, x_test, y_test):

    """Evaluate the model on the test set"""

    return model.evaluate(x_test, y_test)


def plot_model(model, model_fit, epoch_num, x_test, y_test):

    """Plot statistical information of the model and its performance. Based plotting code in Keras official examples: https://keras.io/visualization/"""

    # save an image of the model
    utils.plot_model(model, to_file="model.png", show_shapes=True)

    # accuracy plot (training vs validation)
    plt.plot(model_fit.history["acc"])
    plt.plot(model_fit.history["val_acc"])
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.xlim(0, epoch_num - 1)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend(["Training", "Validation"], loc="lower left")
    plt.savefig("imdb_cnn_accuracy.png")
    plt.close()

    # loss plot (training vs validation)
    plt.plot(model_fit.history["loss"])
    plt.plot(model_fit.history["val_loss"])
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.xlim(0, epoch_num - 1)
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.legend(["Training", "Validation"], loc="upper left")
    plt.savefig("imdb_cnn_loss.png")
    plt.close()

    # classification report
    print("Classification report:")
    y_prediction = model.predict_classes(x_test)
    classes = ["Positive", "Negative"]
    print(classification_report(y_test, y_prediction, target_names=classes))

    # confusion matrix
    matrix = confusion_matrix(y_test, y_prediction)
    print("\nConfusion matrix:\n", matrix)
    data_frame = pd.DataFrame(matrix, index=classes, columns=classes)
    sn.heatmap(data_frame, annot=True, cmap='Blues', fmt='g')
    plt.savefig("imdb_cnn_confusion_matrix.png")


def save_model(model):

    """Save the model weights to a file"""

    model_filename = "model_imdb.json"
    weights_filename = "weights_imdb.txt"
    json = model.to_json()
    with open(model_filename, "w+") as model_file:
            model_file.write(json)
    model.save_weights(weights_filename, overwrite=True)


def main():

    """Main function of the program"""

    # data parameters
    train_fraction = 0.96
    valid_fraction = 0.02
    vocab_size = 10000
    words_per_text = 1000
    top_common_words_to_skip = 0

    # split the data in training, validation and test partitions
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_data_splits(train_fraction, valid_fraction, vocab_size, top_common_words_to_skip, words_per_text)

    # show the splits and data parameters
    print("Data splits:\n\tTraining:\t{}\t({}%)\n\tValidation:\t{}\t({}%)\n\tTest:\t\t{}\t({}%)\n".format(len(x_train), round(train_fraction * 100, 2), len(x_valid), round(valid_fraction * 100, 2), len(x_test), round((1 - train_fraction - valid_fraction)*100, 2)))
    print("Data parameters:\n\tVocabulary size:\t{}\n\tWords per text:\t\t{}\n".format(vocab_size, words_per_text))

    # model parameters
    embedding_dim = 60
    filter_num = 300
    mask_size = 3
    dropout_ratio = 0.3
    hidden_units = 500
    optimizer = "nadam"
    loss_function = "binary_crossentropy"

    # show the model parameters
    print("Model parameters:\n\tEmbeddings dimension:\t{}\n\tConvolution filters:\t{}\n\tConvolution mask:\t\t{}x{}\n\tDropout ratio:\t\t\t{}\n\tHidden units:\t\t\t{}\n\tOptimizer:\t\t\t{}\n\tLoss function:\t\t{}\n".format(embedding_dim, filter_num, mask_size, mask_size, dropout_ratio, hidden_units, optimizer, loss_function))

    # build and compile a neural network model
    model = build_model(embedding_dim, vocab_size, words_per_text, filter_num, mask_size, dropout_ratio, hidden_units, optimizer, loss_function)

    # training parameters
    batch_size = 25
    epoch_num = 3

    # show the training parameters
    print("Training parameters:\n\tBatch size:\t{}\n\tEpochs:\t\t{}\n".format(batch_size, epoch_num))

    # train the model
    model_fit = train_model(model, x_train, y_train, x_valid, y_valid, batch_size, epoch_num)

    # evaluate the performance on test set
    evaluation_result = evaluate_model(model, x_test, y_test)
    print("Evaluation results:\n\tAccuracy:\t{}\n\tLoss:\t\t{}\n".format(evaluation_result[1], evaluation_result[0]))

    # plot model statistics
    plot_model(model, model_fit, epoch_num, x_test, y_test)

    # save the model so it can be loaded later
    save_model(model)


if __name__ == "__main__":
    main()
