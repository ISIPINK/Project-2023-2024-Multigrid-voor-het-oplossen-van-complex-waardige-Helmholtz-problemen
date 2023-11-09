import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import widgets
import numpy as np


def plt_mat(H, title=""):
    plt.imshow(H.toarray())
    plt.colorbar()
    plt.title(title)
    plt.grid(True)
    # plt.show()

# maybe combine with plt_vecs


def plt_mats(mats, titles=[]):
    if len(titles) == 0:
        titles = [""] * len(mats)

    def tmp(i):
        plt_mat(mats[i], titles[i])
    i_slider = widgets.IntSlider(
        value=0, min=0, max=len(mats) - 1, description="i")
    display(widgets.interactive(tmp, i=i_slider))


def plt_vec(v, title=""):
    plt.bar(range(1, len(v)+1), v)
    plt.plot(range(1, len(v)+1), 1.1*v, alpha=1, color="green")
    plt.title(title)
    plt.grid(True)
    # plt.show()


def plt_vecs(vecs, titles=[]):
    if len(titles) == 0:
        titles = [""] * len(vecs)

    def tmp(i):
        plt_vec(vecs[i], titles[i])
    i_slider = widgets.IntSlider(
        value=0, min=0, max=len(vecs) - 1, description="i")
    display(widgets.interactive(tmp, i=i_slider))


def plt_eigen(H, title=""):
    eigenvalues, _ = np.linalg.eig(H.toarray())
    eigenvalues = np.array(sorted(eigenvalues, key=lambda x: np.abs(
        x)*(0 if x.real == 0 else x.real/np.abs(x.real))))
    plt.title(title)
    plt.plot(range(len(eigenvalues)), eigenvalues.real,
             label='Eigenvalues.real')
    plt.plot(range(len(eigenvalues)), eigenvalues.imag,
             label='Eigenvalues.imag')
    plt.plot(range(len(eigenvalues)), np.abs(
        eigenvalues), label='Eigenvalues.abs')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.show()
