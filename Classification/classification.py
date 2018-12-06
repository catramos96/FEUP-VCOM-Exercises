import os
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

def save_fig(fig_id, tight_layout=True):
    path = "images"
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_confusion_matrix(matrix):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    save_fig("confusion_matrix_plot", tight_layout=False)

def main():
    ## load MNIST data
    mnist_raw = loadmat('mnist-original.mat')
    X, y = mnist_raw["data"].T, mnist_raw["label"][0]

    #shuffle randomly
    indexs = np.arange(len(X))

    s = np.arange(indexs.shape[0])
    np.random.shuffle(s)

    X = X[s]
    y = y[s]

    #test and training sets
    train_size = 60000
    test_size = 10000

    train_images = X[:train_size]
    train_labels = y[:train_size]

    test_images = X[train_size:train_size+test_size]
    test_labels = y[train_size:train_size+test_size]

    print "Testing and training sets created!"

    #train a model
    '''
    print "Starting training ..."

    iterations = 1000
    clf = linear_model.SGDClassifier(max_iter = iterations, tol = 1e-3)
    clf.fit(train_images,train_labels)

    print "Training ended!"

    #test model

    print "Predicting ..."

    predicted_labels = clf.predict(test_images)

    print "Prediction ended!"
    print predicted_labels
    '''

    #cross validation
    iterations = 500
    clf = linear_model.SGDClassifier(max_iter = iterations, tol = 1e-3)
    scores = cross_val_score(clf,train_images,train_labels, scoring="accuracy",cv=3)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == '__main__':
    main()