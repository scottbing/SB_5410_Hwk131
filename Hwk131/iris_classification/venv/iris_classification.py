from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import *
import random


def load_data():
    iris = load_iris()

    # convert numpyarr to list get first index of 2
    cutoff = list(iris.target).index(2)
    # throw away data and labels after cutoff
    iris.data = iris.data[:cutoff]
    iris.target = iris.target[:cutoff]

    return iris
# end def load_data():


def sum_row(row):
    # assumes row of length 4 is constant
    # calculate weighted sum
    wsum = row[0] * 0.05 + row[1] * 0.08 + row[2] * 0.75 + row[3] * 1.9
    # return sum(row)
    return wsum


# end def sum_row(row):

def average(data):
    size = data.shape[0]
    summ = 0
    for row in range(size):
        summ += sum_row(data[row])
    return summ / size
# end def average(data):

def test(data, ans, tol):
    sum_res = 0
    for row in data:
        summ = sum_row(row)
        # result = summ + tol > ans and summ - tol < ans
        result = int(summ + tol > ans > summ - tol)
        sum_res += result
    # prints accurate percentage. If showuld be false then 0.0 will be 100%
    print(sum_res / len(data))
# end def test(data, ans,tol):


def main_naive():
    iris = load_data()

    # find where types change
    versic_idx = list(iris.target).index(1)
    # separate all setosa and versic types
    sertosa = iris.data[:versic_idx]
    versic = iris.data[versic_idx:]

    # first 80 percent to train on
    sertosa_train = sertosa[:int(len(sertosa) * .8)]
    # remaining 20% to test with
    sertosa_test = sertosa[int(len(sertosa) * .8):]

    # first 80 percent to train on
    versic_train = versic[:int(len(sertosa) * .8)]
    # remaining 20% to test with
    versic_test = versic[int(len(sertosa) * .8):]

    print("Should be trues")
    test(sertosa_test, average(sertosa_train), 1)
    print("Should be falses")
    test(versic_test, average(sertosa_train), 1)

    print("Should be trues")
    test(versic_test, average(versic_train), 1)
    print("Should be falses")
    test(sertosa_test, average(versic_train), 1)

    print("s ave: ", average(sertosa_train))
    print("v ave: ", average(versic_train))
# end def main_naive():

def main_bkup():
    iris = load_data()

    train_data = iris.data[:int(len(iris.data) * .8)]
    test_data = iris.data[int(len(iris.data) * .8):]
    train_label = iris.target[:int(len(iris.data) * .8)]
    test_label = iris.target[int(len(iris.data) * .8):]

    classifier = Perceptron(learning_rate=0.1)
    classifier.fit(train_data, train_label, 20)
    print("Computed weights are: ", classifier._w)

    for i in range(len(test_label)):
        print(classifier.predict(test_data[i]), test_label[i])
# end def main_bkup():

def scatter_plot(x, y, col, fnames):
    plt.scatter(x,y,c=col)
    plt.title("Iris Dataset Scatterplot")
    plt.xlabel(fnames[0])
    plt.ylabel(fnames[1])
    #plt.show()
#end def scatter_plot(x, y, col, fnames):

def graph_decision(ymin, ymax, x, y, weights, bias):
    w = (bias, weights[0], weights[1])
    xx = np.linspace(ymin, ymax)

    slope = -(w[0]/w[2])/(w[0]/w[1])
    intercept = -w[0]/w[2]

    yy = (slope*xx) + intercept
    plt.scatter(x, y)
    plt.plot(xx, yy, 'k-')
    plt.show()

def best_fit_slope_and_intercept(xs, ys, col, fnames):
    # m = (((mean(xs) * mean(ys)) - mean(np.multiply(xs, ys))) /
    #      ((mean(xs) * mean(xs)) - mean(np.multiply(xs, xs))))

    m = (((mean(xs) * mean(ys)) + mean(np.multiply(xs, ys))) /
         ((mean(xs) * mean(xs)) + mean(np.multiply(xs, xs))))

    # m = (((mean(xs) * mean(ys)) - mean(np.multiply(xs, ys))) /
    #      ((mean(xs) * mean(xs)) - mean(np.multiply(xs, xs))))

    b = mean(ys) - m * mean(xs)

    print(m, b)

    regression_line = [(m*x)+b for x in xs]

    import matplotlib.pyplot as plt
    from matplotlib import style
    style.use('ggplot')

    #plt.scatter(xs,ys)
    plt.scatter(xs, ys, c=col)
    plt.title("Iris Dataset Scatterplot")
    plt.xlabel(fnames[0])
    plt.ylabel(fnames[1])
    plt.plot(xs, regression_line)
    plt.show()


def main():
    iris = load_data()

    versic_idx = list(iris.target).index(1)
    sertosa = iris.data[:versic_idx]
    versic = iris.data[versic_idx:]
    sertosa_label = iris.target[:versic_idx]
    versic_label = iris.target[versic_idx:]

    sertosa_train = sertosa[:int(len(sertosa) * .8)]
    sertosa_test = sertosa[int(len(sertosa) * .8):]
    versic_train = versic[:int(len(versic) * .8)]
    versic_test = versic[int(len(versic) * .8):]

    sertosa_label_train = sertosa_label[:int(len(sertosa) * .8)]
    sertosa_label_test = sertosa_label[int(len(sertosa) * .8):]
    versic_label_train = versic_label[:int(len(versic) * .8)]
    versic_label_test = versic_label[int(len(versic) * .8):]

    train_data = np.concatenate((sertosa_train, versic_train))
    test_data = np.concatenate((sertosa_test, versic_test))
    train_label = np.concatenate((sertosa_label_train, versic_label_train))
    test_label = np.concatenate((sertosa_label_test, versic_label_test))

    #plotting only 2 dimesnsions of the data, the first two vals
    # scatter_plot([val[0] for val in train_data], [val[1] for val in train_data],
    #              train_label, iris.feature_names)

    classifier = Perceptron(learning_rate=0.1)
    classifier.fit(train_data, train_label, 30)
    print("Computed weights are: ", classifier._w)

    # graph_decision(min([val[0] for val in train_data]), max([val[0] for val in train_data]),
    #                [val[0] for val in train_data], [val[0] for val in train_data],
    #                classifier._w, classifier._b)

    best_fit_slope_and_intercept([val[0] for val in train_data], [val[1] for val in train_data],
                                 train_label, iris.feature_names)

    for i in range(len(test_label)):
        print(max(0, classifier.predict(test_data[i])), test_label[i])
# end def main():


if __name__ == "__main__":
    main()
