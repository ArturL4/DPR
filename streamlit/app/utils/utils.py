import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from io import BytesIO


def confusion_matrix_plot(tp, fp, fn, tn, class1, class2, normalize=None, figsize=(15,8)):
    fig, ax = plt.subplots(figsize=figsize)
    conf_mat = np.array([[tp, fp], [fn, tn]], dtype="float")

    if normalize == "true":
        conf_mat /= np.sum(conf_mat, axis=1, dtype="float")

    elif normalize == "pred":
        conf_mat /= np.sum(conf_mat, axis=0, dtype="float")

    elif normalize == "all":
        conf_mat /= np.sum(conf_mat, dtype="float")

    conf_matrix = sns.heatmap(
        conf_mat,
        annot=True,
        yticklabels=[f"{class1}", f"{class2}"],
        xticklabels=[f"{class1}", f"{class2}"],
        cmap="crest",
    )
    conf_matrix.set(xlabel=r"True class $\omega$", ylabel=r"Prediction $\^{\omega}$")
    return fig, conf_mat
