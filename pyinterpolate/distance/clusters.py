from typing import Union, List

import hdbscan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def aggregate_cluster(ds: np.ndarray, labels: np.ndarray, cluster: int, method='mean') -> np.ndarray:
    """
    Function aggregates points in a cluster into a single point.

    Parameters
    ----------
    ds : numpy ndarray
        Points(x, y, value)

    labels : numpy ndarray
        Cluster labels (integers). The length of this array must be equal to the length of ``ds`` points.

    cluster : int
        Number of cluster to aggregate.

    method : str, default='mean'
        The method of aggregation, available methods are:
        * ``mean``,
        * ``median``,
        * ``sum``

    Returns
    -------
    declustered : numpy ndarray
        Declustered list of points.
    """

    # Select points with a specific label
    points = ds[labels == cluster]

    # Calculate their centroid
    centroid = np.mean(points[:, :1], axis=0)

    # Calculate metrics
    if method == 'mean':
        metric = np.mean(points[:, 2])
    elif method == 'median':
        metric = np.median(points[:, 2])
    elif method == 'sum':
        metric = np.sum(points[:, 2])
    else:
        raise KeyError('Unknown aggregation method. Available methods: "mean", "median", "sum".')

    new_point = np.zeros(3)
    new_point[:1] = centroid
    new_point[-1] = metric

    output = ds[~(labels == cluster)].copy()

    declustered = np.append(output, [new_point], axis=0)
    return declustered


class ClusterDetector:
    """
    Class detects spatial clusters in data and allows to remove those from a dataset.

    Parameters
    ----------
    verbose : bool, default = True
        Should print process information?


    Attributes
    ----------
    ds : numpy ndarray
        Input points (x, y, value)

    aggregated : numpy ndarray, default=None
        Transformed (aggregated) points.

    clusterer : HDBSCAN
        Clustering class derived from hdbscan package (see References).

    verbose : bool, default=True
        Print process steps.

    Methods
    -------
    aggregate_clusters()
        Clean and aggregate clustered data.

    fit_clusters()
        Train and fit HDBSCAN model.

    get_labels()
        Gets cluster label of each point.

    show_clusters()
        Shows scatterplot with clustered points and their labels.

    Raises
    ------
    RuntimeError
        Model is not fitted and user tries to use ``aggregate_clusters()``, ``get_labels()`` or ``show_clusters()``
        methods.

    References
    ----------
    Pyinterpolate uses HDBSCAN package: https://hdbscan.readthedocs.io/en/latest/index.html

    [1] L. McInnes, J. Healy, S. Astels, hdbscan: Hierarchical density based clustering.
    In: Journal of Open Source Software, The Open Journal, volume 2, number 11. 2017

    """

    def __init__(self, ds, verbose=True):
        self.ds = ds
        self.aggregated = None
        self.clusterer = None
        self.verbose = verbose

    def aggregate_clusters(self,
                           cluster_labels: Union[int, List],
                           method="mean",
                           store_aggregated=False) -> np.ndarray:
        """
        Method aggregates clusters

        Parameters
        ----------
        cluster_labels : int or List
            Cluster label or multiple labels to transform dataset.

        method : str, default='mean'
            The method of aggregation, available methods are:
            * ``mean``,
            * ``median``,
            * ``sum``

        store_aggregated : bool
            Should class store aggregated and declustered dataset.

        Returns
        -------
        aggregated : numpy ndarray
            Transformed (declustered) dataset.
        """
        if self.clusterer is None:
            raise RuntimeError('You must fit model before declustering')

        if isinstance(cluster_labels, int):
            cluster_labels = [cluster_labels]

        aggregated = self.ds.copy()

        for clabel in cluster_labels:
            if self.verbose:
                print('Removing cluster number', clabel)

            aggregated = aggregate_cluster(
                ds=aggregated,
                labels=self.get_labels(),
                cluster=clabel,
                method=method
            )

        if store_aggregated:
            self.aggregated = aggregated

        return aggregated

    def fit_clusters(self,
                     min_cluster_size=5,
                     min_samples=None,
                     max_cluster_size=0,
                     cluster_selection_epsilon=0.0,
                     cluster_selection_method='leaf',
                     **kwargs):
        """
        Method detects clusters in a data. It uses DBSCAN algorithm described here:
        https://hdbscan.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        min_cluster_size : int, default=5
            The minimum size of clusters.

        min_samples : int, optional
            The number of samples in a neighbourhood for a point to be considered a core point.

        max_cluster_size : int, default=0
            The maximum number of points within a cluster,

        cluster_selection_epsilon : float, default=0, range(0, 1)
            A distance threshold. Clusters below this value will be merged.

        cluster_selection_method : str, default="leaf"
            The method used to select clusters from the condensed tree. Available methods: "eof" or "leaf"

        kwargs : Any
            Other parameters that could be passed into clusterer. The full list is available here:
            https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan
        """

        params = {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "max_cluster_size": max_cluster_size,
            "cluster_selection_method": cluster_selection_method,
            "cluster_selection_epsilon": cluster_selection_epsilon
        }

        if kwargs:
            params.update(kwargs)

        if self.verbose:
            print('Model fit...')

        self.clusterer = hdbscan.HDBSCAN(
            **params
        )

        self.clusterer.fit(self.ds[:, :1])

        if self.verbose:
            print('Clusters detected, number of clusters:', sum(self.clusterer.labels_ >= 0))

    def get_labels(self) -> np.ndarray:
        """
        Method returns cluster labels.

        Returns
        -------
        labels : numpy ndarray
        """
        if self.clusterer is None:
            raise RuntimeError('You must fit model to get labels')

        return self.clusterer.labels_

    def show_clusters(self):
        """
        Method shows clusters in a data.
        """
        if self.clusterer is None:
            raise RuntimeError('You must fit model before visualization')

        # Prepare labels and colormap
        lbls = self.clusterer.labels_
        no_of_labels = len(np.unique(lbls))

        # Define colormap
        color = mpl.colormaps['summer']
        clist = [color(i) for i in range(color.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Clusters cmap', clist, color.N)
        bounds = np.linspace(0, no_of_labels, no_of_labels + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # Plot
        fig, ax = plt.subplots(1, 1)
        scat = ax.scatter(self.ds[:, 0], self.ds[:, 1], c=lbls, cmap=cmap, norm=norm, edgecolor='black')
        cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
        cb.set_label('Labels')

        # Annotate
        for i, txt in enumerate(lbls):
            ax.annotate(txt, (self.ds[:, 0][i], self.ds[:, 0][i]))

        plt.show()
