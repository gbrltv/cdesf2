from typing import List
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def cumulative_stream_drifts(stream_length: int, drifts_index: List, save_path: str) -> None:
    """
    Plots the cumulative number of drifts and vertical lines indicating where drifts were detected

    Parameters
    --------------------------------------
    stream_length: int
        Size of the stream used to set the x axis
    drifts_index: List
        Index points where drifts were detected in the stream
    save_path: str
        Path for the file to be saved
    """

    plt.figure(figsize=(15, 7.5))

    if len(drifts_index) >= 1:
        # Only plot if there is a drift.
        for drift in drifts_index[:-1]:
            plt.axvline(drift, ymin=0, ymax=1, linestyle='dashed', c='k', alpha=0.5)
        plt.axvline(drifts_index[-1], ymin=0, ymax=1, linestyle='dashed', c='k', alpha=0.5, label='Drift points')

        y_values = list(range(1, len(drifts_index)+1))
        drifts_index.append(stream_length)
        y_values.append(y_values[-1])
        ax = sns.lineplot(x=drifts_index, y=y_values, linewidth=3, label='Number of drifts')

    plt.xlim(0, stream_length)
    plt.title("Drifts and cumulative number of drifts in the stream", size=20)
    plt.xlabel("Event stream", size=15)
    plt.ylabel("Number of drifts", size=15)
    plt.xticks(size=12.5)
    plt.yticks(size=12.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

