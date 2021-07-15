from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ..data_structures import Cluster


def feature_space(process_name: str,
                  event_index: int,
                  cp: int,
                  normals: np.array,
                  outliers: np.array,
                  normal_clusters: Tuple[List[List], List[List]],
                  o_clusters: List[Cluster],
                  epsilon: float,
                  plot_path: str):
    """
    Plot all types of clusters in a current point in the stream.
    Current active cases are also plotted.
    Saves the plot according to the plot_path attribute.

    Parameters
    --------------------------------------
    process_name: str
        Name of the process
    event_index: int
        Current index in the stream
    cp: int
        Current check point
    normals: np.array
        Normal cases
    outliers: np.array
        Anomalous cases
    normal_clusters: Tuple[List[List], List[List]]
        Core and potential micro-clusters maintained by denstream
    o_clusters: List[Cluster]
        Outlier micro-clusters maintained by denstream
    epsilon: float
        Defines the maximum range of a micro-cluster action
    plot_path: str
        Path for the file to be saved
    """
    c_clusters, p_clusters = normal_clusters[0], normal_clusters[1]

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    for group in p_clusters:
        for cluster in group:
            ax.add_patch(patches.Circle(cluster.centroid, cluster.radius, fill=False, color='blue', ls=':'))
            ax.add_patch(patches.Circle(cluster.centroid, epsilon, fill=False, color='blue', ls='--'))
            ax.annotate(f'{cluster.id}', xy=cluster.centroid, color='red')

    for cluster in o_clusters:
        ax.add_patch(patches.Circle(cluster.centroid, cluster.radius, fill=False, color='red', ls='--'))
        ax.add_patch(patches.Circle(cluster.centroid, epsilon, fill=False, color='red', ls='--'))
        ax.annotate(f'{cluster.id}', xy=cluster.centroid, color='red')

    for group in c_clusters:
        for cluster in group:
            ax.add_patch(patches.Circle(cluster.centroid, cluster.radius, fill=False, color='black'))
            ax.add_patch(patches.Circle(cluster.centroid, epsilon, fill=False, color='black', ls='--'))
            ax.annotate(f'{cluster.id}', xy=cluster.centroid, color='red')

    for p in normals:
        ax.scatter(*p, color='black', marker='o', s=15)

    for p in outliers:
        ax.scatter(*p, color='orange', marker='x', s=15)

    plt.title(f'{process_name}, stream index:{event_index}, CP: {cp}')
    plt.xlabel('$GD_{trace}$', size=15)
    plt.ylabel('$GD_{time}$', size=15)
    plt.tight_layout()
    plt.savefig(f'{plot_path}/{event_index}.png')
    plt.close()
