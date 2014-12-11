from personal_classifier import PersonalClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class RandomForrest(PersonalClassifier):
    def __init__(self, featured_frames):
        super(RandomForrest, self).__init__(featured_frames)

    def test(self):
        iris = load_iris()
        X = iris.data
        Y = iris.target

        h = .02  # step size in the mesh

        clf = RandomForestClassifier(n_estimators=10)
        clf = clf.fit(X, Y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(4, 3))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.savefig('img/rndforrest.png')

