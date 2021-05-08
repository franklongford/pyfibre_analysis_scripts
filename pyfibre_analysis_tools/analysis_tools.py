import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's
        radii.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = mpl.patches.Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor, **kwargs
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ax.plot([mean_x], [mean_y], '*',
            color='yellow', markersize=20,
            markeredgecolor='black')

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def scatter(array, colors, sizes,
            ellipse=True, marker='o', alpha=0.7,
            fig=None, ax=None, cb=None):

    N = np.unique(colors).size
    max_n = np.max(colors)
    min_n = np.min(colors)
    cmap = plt.cm.jet
    bounds = np.linspace(1, max_n, max_n+1)

    if N > 1:
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    else:
        norm = None

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    scat = ax.scatter(
        array[:, 0], array[:, 1], c=colors, s=sizes*80,
        alpha=alpha, cmap=cmap, norm=norm, marker=marker)

    if ellipse:
        for label in range(min_n, max_n+1):
            indices = np.where(colors == label)
            values = array[indices]
            confidence_ellipse(values[:, 0], values[:, 1], ax, n_std=1.0, 
                               edgecolor='red', linestyle='dashed')

    if cb is None:
        cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)

    return fig, ax, cb


def load_databases(filename, data_directories, ext=None):

    # Define an empty database to load the files into
    database = pd.DataFrame()

    if ext is not None and not filename.endswith(ext):
        filename += ext

    def _load_db(db_path, group, label):
        """Helper function to load a database and assign group + label
        columns
        """
        if db_path.endswith('.h5'):
            db = pd.read_hdf(db_path, key='df')
        elif db_path.endswith('.xls'):
            db = pd.read_excel(db_path, key='df')
        db['Group'] = group
        db['Label'] = label
        return db

    print("{:20} | {:10} | {:10}".format('Group', 'N', 'Label'))
    print("-" * 42)
    # Loop through the directories to load each database
    for i, directory in enumerate(data_directories):
        # The name of the folder becomes the name of the group
        group = os.path.split(directory)[-1].lower()

        try:
            # Try to load Pandas Dataframe from directory
            db_path = os.path.join(directory, filename)
            db = _load_db(db_path, group, i + 1)
            database = pd.concat([database, db])
        except IOError:
            # Look in sub directories if not database file is present
            for folder in os.listdir(directory):
                db_path = os.path.join(directory, folder, filename)
                db = _load_db(db_path, group, i + 1)
                database = pd.concat([database, db])

        print("{:<20} | {:<10} | {:<10}".format(
            group, len(database['Group'] == group), i+1))

    return database
