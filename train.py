from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

# Fit a model
for depth in range(2, 5):
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(acc)
    # Plot it
    disp = ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, normalize="true", cmap=plt.cm.Blues
    )
    plotfile = f"plot-{depth}.png"
    plt.savefig(plotfile)
    with open("report.md", "a") as outfile:
        outfile.write(f"# Forest depth: {depth}\n\n")
        outfile.write(f"Accuracy: {acc}\n")
        outfile.write(f"![Confusion Matrix]({plotfile})\n\n\n")
