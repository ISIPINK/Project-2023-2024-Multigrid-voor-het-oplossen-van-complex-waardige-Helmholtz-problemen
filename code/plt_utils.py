from bokeh.io import output_notebook, show
from bokeh.models import HoverTool, WheelZoomTool, PanTool, ResetTool, ColumnDataSource
from bokeh.plotting import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
import numpy as np
import pacmap
from scipy.linalg import norm


def plt_mat(H, title=""):
    plt.imshow(H.toarray())
    plt.colorbar()
    plt.title(title)
    plt.grid(True)
    print(f"shape={H.shape}")


def plt_mats(mats, titles=[]):
    if len(titles) == 0:
        titles = [""] * len(mats)

    @interact(
        i=widgets.IntSlider(value=0, min=0, max=len(mats) - 1, description="i")
    )
    def tmp(i):
        plt_mat(mats[i], titles[i])


def plt_vec(v, title=""):
    plt.bar(range(1, len(v)+1), v.real)
    dv = np.diff(v)
    if norm(dv) < 0.2*norm(v):
        plt.plot(range(1, len(v)+1), 1.1*v.real, alpha=1, color="green")
    plt.title(title)
    plt.grid(True)


def plt_vecs(vecs, titles=[]):
    if len(titles) == 0:
        titles = [""] * len(vecs)

    @interact(
        i=widgets.IntSlider(value=0, min=0, max=len(vecs) - 1, description="i")
    )
    def tmp(i):
        plt_vec(vecs[i], titles[i])


def plt_vec2D(vector, title=""):
    d = int(np.sqrt(len(vector)))
    matrix = np.reshape(vector, (d, d), order='F')
    plt.figure(figsize=(13, 5))
    # Plot for the real part
    plt.subplot(1, 2, 1)
    plt.imshow(matrix.real, cmap='viridis')
    plt.colorbar()
    plt.title(f"Real Part - {title}")

    # Plot for the imaginary part
    plt.subplot(1, 2, 2)
    plt.imshow(matrix.imag, cmap='viridis')
    plt.colorbar()
    plt.title(f"Imaginary Part - {title}")
    plt.show()


def plt_vecs2D(vecs, titles=[]):
    if len(titles) == 0:
        titles = [""] * len(vecs)

    @interact(
        i=widgets.IntSlider(value=0, min=0, max=len(vecs) - 1, description="i")
    )
    def tmp(i):
        plt_vec2D(vecs[i], titles[i])


def plt_vecs_pacmap(vecs):
    reducer = pacmap.PaCMAP()
    embedding = reducer.fit_transform(vecs)

    output_notebook()

    source = ColumnDataSource(data={
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "eigvec": vecs
    })

    p = figure(tools=[WheelZoomTool(), PanTool(), ResetTool()])
    p.scatter("x", "y", size=10, source=source)

    hover = HoverTool(tooltips=[
        ("Index", "$index"),
        ("Eigenvector", "@eigvec")
    ])

    p.add_tools(hover)
    show(p)


def plt_R_im_abs(xs, ys, label, linestyle=None):
    plt.plot(xs, ys.real, label=f'R({label})', linestyle=linestyle)
    plt.plot(xs, ys.imag, label=f'Im({label})', linestyle=linestyle)
    plt.plot(xs, np.abs(ys), label=f'|{label}|', linestyle=linestyle)


def plt_eigen(Hs, labels, title=""):
    for H, label in zip(Hs, labels):
        eigenvalues, _ = np.linalg.eig(H.toarray())
        eigenvalues = np.array(sorted(eigenvalues, key=lambda x: np.abs(
            x)*(0 if x.real == 0 else x.real/np.abs(x.real))))
        xs = np.linspace(0, 1, len(eigenvalues))
        plt_R_im_abs(xs, eigenvalues, f"eig({label})")

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.show()
