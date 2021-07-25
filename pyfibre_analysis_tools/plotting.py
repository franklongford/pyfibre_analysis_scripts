import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from sklearn.metrics import auc


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


def plot_lda_analysis(lda, training, test, columns, tick_labels, tag=''):
    # Plotting the training and test sets for an example LDA
    X_plot_train = lda.transform(training[0])
    X_plot_test = lda.transform(test[0])

    plt.figure(0)
    fig, ax, cb = scatter(X_plot_train, training[1], np.ones(training[1].shape))
    fig, ax, cb = scatter(X_plot_test, test[1], np.ones(test[1].shape),
                          ellipse=False, marker='x', alpha=0.6, fig=fig, ax=ax, cb=cb)
    cb.set_ticklabels(tick_labels)
    # plt.axis('off')
    # plt.axis([-5, 5, -5, 5])
    plt.tight_layout()
    plt.savefig(f'lda_{tag}.png')
    plt.show()

    # Plot LDA coefficients
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    im = ax.matshow(lda.coef_, interpolation='none', cmap='coolwarm')
    fig.colorbar(im)

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(list(columns))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_yticklabels(tick_labels)

    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
             ha="left", va="center", rotation_mode="anchor")
    plt.show()

    # Plotting metric covariance
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    im = ax.matshow(lda.covariance_, interpolation='none', cmap='coolwarm')
    fig.colorbar(im)

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(list(columns))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_yticklabels(list(columns))

    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
             ha="left", va="center", rotation_mode="anchor")
    plt.show()


def plot_roc_curve(tprs, aucs, tag=''):
    """Creates a ROC figure to display from a set of true positive
    result and area under the curve values

    Parameters
    ----------
    tprs, aucs: array-like
        True positive rates and area under the curve results
    """
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_fpr = np.linspace(0, 1, 100)

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic example")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_{tag}.png')
    plt.show()
