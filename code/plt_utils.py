import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import interact, widgets
import numpy as np


def plt_mat(H, title=""):
    plt.imshow(H.toarray())
    plt.colorbar()
    plt.title(title)
    plt.show()


def plt_vec(v, title=""):
    plt.bar(range(len(v)), v)
    plt.title(title)
    plt.show()


def plt_vecs(vecs, titles=[]):
    if len(titles) == 0:
        titles = [""] * len(vecs)

    def tmp(i):
        plt_vec(vecs[i], titles[i])
    i_slider = widgets.IntSlider(
        value=0, min=0, max=len(vecs) - 1, description="i")
    display(interact(tmp, i=i_slider))
