import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import interact, widgets
import numpy as np


def plt_mat(H, title=""):
    plt.imshow(H.toarray())
    plt.colorbar()
    plt.title(title)
    plt.show()

# maybe combine with plt_vecs


def plt_mats(mats, titles=[]):
    if len(titles) == 0:
        titles = [""] * len(mats)

    def tmp(i):
        plt_mat(mats[i], titles[i])
    i_slider = widgets.IntSlider(
        value=0, min=0, max=len(mats) - 1, description="i")
    display(interact(tmp, i=i_slider))


def plt_vec(v, title=""):
    plt.bar(range(1, len(v)+1), v)
    plt.plot(range(1, len(v)+1), 1.1*v, alpha=1, color="green")
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
