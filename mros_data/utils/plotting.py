from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from librosa.display import specshow
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper right')


def plot_data(
    data: np.ndarray,
    events: np.ndarray,
    fs: int,
    predictions: Optional[np.ndarray] = None,
    channel_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    ind: Optional[int] = None,
) -> Tuple[Figure, Axes]:

    # Get current axes or create new
    if ax is None:
        fig, (ax2, ax) = plt.subplots(2, figsize=((24,9)))
        ax.set_xlabel("Time (s)")
    else:
        fig = ax.get_figure()
        ax.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off
        
    font = {'family': 'Times new roman',
            'weight': 'bold',
            'size': 20}

    plt.rc('font', **font)


    C, T = data.shape
    time_vector = np.arange(T) / fs

    assert (
        len(channel_names) == C
    ), f"Channel names are inconsistent with number of channels in data. Received {channel_names=} and {data.shape=}"

    event_label_dict = {0.0: 'Arousal', 1.0: 'Leg Movement', 2.0: 'Sleep-disordered Breathing'}
    # Plot events
    if True:
        for event_label in np.unique(events[:, -1]):
            if event_label == 0.0:
                color = "r"
            elif event_label == 1.0:
                color = "yellow"
            elif event_label == 2.0:
                color = "cornflowerblue"
            class_events = events[events[:, -1] == event_label, :-1] * T / fs
            for evt_start, evt_stop in class_events:
                label = np.unique(event_label_dict[event_label])
                ax2.axvspan(evt_start, evt_stop, facecolor=color, edgecolor = None, alpha=0.6, label=label)
                legend_without_duplicate_labels(ax2)


    if True:
        for event_label in np.unique(predictions[:, -1]):
            if event_label == 0.0:
                color = "maroon"
            elif event_label == 1.0:
                color = "goldenrod"
            elif event_label == 2.0:
                color = "royalblue"
            else:
                color = 'gray'
            class_events = predictions[predictions[:, -1] == event_label, :-1] * T / fs
            for evt_start, evt_stop in class_events:
                label = np.unique(event_label_dict[event_label])
                ax.axvspan(evt_start, evt_stop, facecolor=color, edgecolor = None, alpha=0.6, label=label)
                legend_without_duplicate_labels(ax)

    # Calculate the offset between signals
    data = (
        2
        * (data - data.min(axis=-1, keepdims=True))
        / (data.max(axis=-1, keepdims=True) - data.min(axis=-1, keepdims=True))
        - 1
    )
    offset = np.zeros((C, T))
    for idx in range(C - 1):
        # offset[idx + 1] = -(np.abs(np.min(data[idx])) + np.abs(np.max(data[idx + 1])))
        offset[idx + 1] = -2 * (idx + 1)

    # Plot signals
    ax.plot(time_vector, data.T + offset.T, color="gray", linewidth=1)
    ax2.plot(time_vector, data.T + offset.T, color="gray", linewidth=1)
    # Adjust plot visuals
    ax.set_xlim(time_vector[0], time_vector[-1])
    ax2.set_xlim(time_vector[0], time_vector[-1])
    ax.set_yticks(ticks=offset[:, 0], labels=channel_names)
    ax2.set_yticks(ticks=offset[:, 0], labels=channel_names)
    ax2.set_title(title)
    plt.savefig('D:/predictions/high_complex_all_predictions/high_complex_labels_{index}.png'.format(index = str(ind)), dpi = 200)
    return fig, ax


def plot_spectrogram(
    data: np.ndarray,
    fs: int,
    step_size: int,
    window_length: int,
    nfft: int,
    ax: Axes = None,
    display_type: str = "hz",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> Tuple[Figure, Axes]:

    # Get current axes or create new
    if ax is None:
        fig, ax = plt.subplots(figsize=(25, 4))
    else:
        fig = ax.get_figure()

    specshow(
        data,
        sr=fs,
        hop_length=step_size,
        win_length=window_length,
        n_fft=nfft,
        y_axis=display_type,
        x_axis="time",
        ax=ax,
        fmin=fmin,
        fmax=fmax,cmap='magma'
    )

    return fig, ax
