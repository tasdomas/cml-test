import json
import os
from textwrap import dedent
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

best_depth = 2
best_accuracy = 0
best_estimator = None

# Fit a model
for depth in range(2, 5):
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    if acc > best_accuracy:
        best_depth = depth
        best_estimator = clf
        best_accuracy = acc

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, normalize="true", cmap=plt.cm.Blues
    )
    plotfile = f"plot-{depth}.png"
    plt.savefig(plotfile)
    with open("report.md", "a") as outfile:
        outfile.write(dedent(f'''
          # Forest depth: {depth}

          Accuracy: {acc}
          ![Confusion Matrix]({plotfile})

        '''))

    # Update final report
    with open("final-report.md", "w") as report:
        disp = ConfusionMatrixDisplay.from_estimator(
            best_estimator, X_test, y_test, normalize="true", cmap=plt.cm.Blues
        )
        plt.savefig("plot-best.png")

        report.write(dedent(f'''
        # Best result:

        Depth: {best_depth}
        Accuracy: {best_accuracy}
        ![Confusion Matrix](./plot-best.png)
        '''))

    # Pretend it's a lengthy training.
    time.sleep(5)
