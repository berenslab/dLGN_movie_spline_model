import ast
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import re
import scipy
import seaborn as sns
import base64
import json

def get_color_dict(palette_name='colorblind', n=None, color_names=None, add_default_colors=True):
    """
    Returns color map dictionary from seaborn color palettes with hexadecimal triplet values.

    Parameters
    ----------
    palette_name : str
        Name of palette. Possible values are:
        'colorblind', 'deep', 'muted', 'bright', 'pastel', 'dark'
        For more options see seaborn.color_palette() docs.
    n : int, optional
        Number of colors. If not None and color_names is given, n must be equal to len(color_names).
    color_names : list, optional
        List of color names as strings e.g. for the color palette 'colorblind': ['blue',
        'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'yellow', 'lightblue'].
        Must have the same length as colors in the color palette. If None, color names are set to ['c0', 'c1', 'c2', ...].
    add_default_colors : bool, optional
        Add default colors that are needed but are not part of the color palette such as 'black'.
        If set to False, the color dict will only contain colors of the chosen palette.

    Returns
    -------
    color_dict: dict
        Color name as key and color triplet as value.
    """
    
    # Get list of sns colors as hexadecimal triplet
    sns_colors_l = sns.color_palette(palette_name).as_hex()
    if n is not None:
        sns_colors_l = sns_colors_l[:n]
    # Define color names if none are given
    if color_names is None:
        color_names = ["c%d" % i for i in range(0, len(sns_colors_l))]
    # Create color dict
    color_dict = dict.fromkeys(color_names)
    for color_name, color in zip(color_names, sns_colors_l):
        color_dict[color_name] = color
    # Add additional colors to palette
    if add_default_colors is True:
        color_dict['black'] = '#000000'

    return color_dict

def plot_fr_scatter(x, y, labels, facecolor, axlims, figsize, plot_pval=True, s=20, marker='.', mark_neurons=None):
    """ Returns figure of firing rate scatter.
    
    Parameters
    ----------
    x: np.array
        x-values
    y: np.array
        y-values
    labels: list
        ['xlabel', ylabel']
    facecolor: str
        Color of markers
    axlims: list
        [min, max] of axis
    figsize: np.array
        (height, width) of figure
        
    Returns
    -------
    fig:
        figure
    ax_dict: dict
        Contains all axis
    """
    # Get p-value
    W, p = scipy.stats.wilcoxon(x, y)
    print('p = {:.2E}'.format(p))
    print('{:.2f} vs. {:.2f} spikes per sec'.format(np.mean(x), np.mean(y)))

    # Initialize figure
    with plt.style.context("matplotlib_config.txt"):
        mosaic = """
        a
        """
        fig, ax_dict = plt.subplot_mosaic(
            mosaic,
            figsize=figsize,
            dpi=200,
            constrained_layout=True,
            #sharex=True,
        )

        txt_kwargs = dict(
            color="black", fontsize="larger"
        )
        ax = ax_dict['a']
        ax.scatter(x, y, facecolor=facecolor, edgecolor='none', alpha=.5, s=s, marker=marker)
        if mark_neurons is not None:
            for idx_neuron in mark_neurons:
                ax.scatter(x[idx_neuron], y[idx_neuron], color='k', s=5, marker='x')
        ax.plot([0, axlims[1]], [0, axlims[1]], color='grey', linestyle='--', zorder=-1)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_xlim(axlims)
        ax.set_ylim(axlims)
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_yticks(xticks)
        sns.despine(offset=5, trim=False, ax=ax)

        # Add p-value and n
        if plot_pval:
            if p<0.05:
                txt = 'p < 0.05'
            if p<0.01:
                txt = 'p < 0.01'
            if p<0.001:
                txt = 'p < 0.001'
            else:
                f'p = {np.round(p,3)}'
            txt = txt + f'\nn = {len(x)}'
            #txt = (('p < 0.001' if p<0.001 else f'p = {np.round(p,3)}') + '\n' + f'n = {len(x)}')
            ax.text(0, 1, txt, va='top', fontsize=6, transform=ax.transAxes)

        # Plot inset histogram of differences
        ax_ins = inset_axes(ax, width="30%", height="30%", loc=4, borderpad=0.75)
        ax_ins.hist(np.log2(y/x), color=facecolor, alpha=0.5)  # log2 fold change
        ax_ins.patch.set_facecolor('none')
        ax_ins.spines['left'].set_visible(False)
        ax_ins.spines['right'].set_visible(False)
        ax_ins.tick_params(bottom=True, left=False, right=True,
                           labelbottom=True, labelleft=False, labelright=True,
                           length=1.5, pad=1, labelsize=5)
    
    return fig, ax_dict

def adjust_spines(ax, spines, spine_pos=5, color='k', linewidth=None, smart_bounds=True):
    """
    Convenience function to adjust plot axis spines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to adjust.
    spines : list or None
        List of spines to adjust (e.g. ['left', 'bottom']). If None, all spines are made invisible.
    spine_pos : int or float, optional
        Position of the spines. Default is 5.
    color : str, optional
        Color of the spines. Default is 'k' (black).
    linewidth : float, optional
        Line width of the spines. If None, uses default.
    smart_bounds : bool, optional
        Use smart bounds. Default is True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The adjusted axes.
    """

    # If no spines are given, make everything invisible
    if spines is None:
        ax.axis('off')
        return

    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', spine_pos))  # outward by x points
            spine.set_color(color)
            if linewidth is not None:
                spine.set_linewidth = linewidth
        else:
            spine.set_visible(False)  # make spine invisible

    # Turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # No visible yaxis ticks and tick labels
        plt.setp(ax.get_yticklabels(), visible=False)  # hides ticklabels but not ticks
        plt.setp(ax.yaxis.get_ticklines(), color='none')  # changes tick color to none

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # No visible xaxis ticks and tick labels
        plt.setp(ax.get_xticklabels(), visible=False)  # hides ticklabels but not ticks
        plt.setp(ax.xaxis.get_ticklines(), color='none')  # changes tick color to none
    
    return ax

def get_t_half_max_to_peak(kernel, kernel_tv):
    """
    Calculate the time from half-max to peak of a kernel.

    Parameters
    ----------
    kernel : np.array
        The kernel values.
    kernel_tv : np.array
        The time values corresponding to the kernel.

    Returns
    -------
    time_half_peak_to_peak : float
        Time from half-max to peak in milliseconds.
    """
    peak_val = np.max(kernel)
    half_peak_val = peak_val / 2
    idx_peak = np.where(kernel == peak_val)[0][0]
    idx_half_peak = np.argmin(abs(kernel[:idx_peak] - half_peak_val))
    peak_time = kernel_tv[idx_peak]
    half_peak_time = kernel_tv[idx_half_peak]
    time_half_peak_to_peak = np.diff([half_peak_time, peak_time])[0]  # ms

    return time_half_peak_to_peak

def get_formatted_pval(p, stars=False, show_ns_val=True):
    """
    Format p-value for display.

    Parameters
    ----------
    p : float
        The p-value to format.
    stars : bool, optional
        If True, returns stars instead of numerical values. Default is False.
    show_ns_val : bool, optional
        If True, shows 'p=...' for non-significant p-values. Default is True.

    Returns
    -------
    p_formatted : str
        Formatted p-value.
    """
    
    if 1.00e-02 < p <= 5.00e-02:
        if stars:
            p_formatted = '*'
        else:
            p_formatted = 'p<0.05'
    elif 1.00e-03 < p <= 1.00e-02:
        if stars:
            p_formatted = '**'
        else:
            p_formatted = 'p<0.01'
    elif 1.00e-04 < p <= 1.00e-03:
        if stars:
            p_formatted = '***'
        else:
            p_formatted = 'p<0.001'
    elif p <= 1.00e-04:
        if stars:
            p_formatted = '****'
        else:
            p_formatted = 'p<0.0001'
    else:
        if show_ns_val:
            p_formatted = 'p={:.3f}'.format(p)
        else:
            p_formatted = 'ns'
    return p_formatted

def find_keys_with_p(keys, ps, p_thr=0.01, p_lower_limit = 0.0):
    """
    Find keys with p-values within specified thresholds.

    Parameters
    ----------
    keys : list
        List of keys.
    ps : list
        List of p-values corresponding to the keys.
    p_thr : float, optional
        Upper threshold for p-values. Default is 0.01.
    p_lower_limit : float, optional
        Lower limit for p-values. Default is 0.0.

    Returns
    -------
    inds : np.array
        Indices of keys that meet the criteria.
    filtered_ps : np.array
        Filtered p-values that meet the criteria.
    filtered_keys : np.array
        Filtered keys that meet the criteria.
    """
    
    inds_thr = np.where(np.array(ps) <= p_thr)[0]
    if p_lower_limit != 0.0:
        inds_lower_limit = np.where(np.array(ps) > p_lower_limit)[0]
        inds = np.intersect1d(inds_thr, inds_lower_limit)
    else:
        inds = inds_thr
        
    filtered_ps = np.array(ps)[inds]
    filtered_keys = np.array(keys)[inds]
    
    return inds, filtered_ps, filtered_keys


def find_keys_within_range(keys, omis, omi_lower, omi_upper):
    """
    Find keys within a specified range of values.

    Parameters
    ----------
    keys : list
        List of keys.
    omis : list
        List of values corresponding to the keys.
    omi_lower : float
        Lower bound of the range.
    omi_upper : float
        Upper bound of the range.

    Returns
    -------
    inds : np.array
        Indices of keys that meet the criteria.
    filtered_omis : np.array
        Filtered values that meet the criteria.
    filtered_keys : np.array
        Filtered keys that meet the criteria.
    """
    
    inds_lower = np.where(np.array(omis) > omi_lower)[0]
    inds_upper = np.where(np.array(omis) <= omi_upper)[0]
    inds = np.intersect1d(inds_lower, inds_upper)
    
    filtered_omis = np.array(omis)[inds]
    filtered_keys = np.array(keys)[inds]
    
    return inds, filtered_omis, filtered_keys


def find_neurons_per_bins(firing_rates, mabin, nextbin):
    """
    Find indices of neurons with firing rates within specified bins.

    Parameters
    ----------
    firing_rates : list or np.array
        List of firing rates.
    mabin : float
        Lower bound of the bin.
    nextbin : float
        Upper bound of the bin.

    Returns
    -------
    inds_within_bins : np.array
        Indices of neurons within the specified bins.
    """
    
    inds = np.where(firing_rates >= mabin)[0]
    inds2 = np.where(firing_rates <= nextbin)[0]
    
    inds_within_bins = np.intersect1d(inds2, inds)
    
    return inds_within_bins

def preprocess_array_string(s):
    """
    Preprocess a string for array conversion.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    s : str
        Processed string.
    """
    # Remove any newline characters
    s = s.replace('\n', ' ')
    # Ensure commas between numbers using a more robust regular expression
    s = re.sub(r'(?<=\d)\s+(?=-?\d)', ', ', s)
    return s

def str_to_array(s):
    """
    Convert a string representation of an array to a NumPy array.

    Parameters
    ----------
    s : str or np.ndarray
        Input string or numpy array.

    Returns
    -------
    np.array
        Converted NumPy array.

    Raises
    ------
    ValueError
        If the input is not a string or numpy array.
    """
    
    if isinstance(s, np.ndarray):
        return s  # If it's already a NumPy array, return it as is
    
    if not isinstance(s, str):
        raise ValueError(f"Input must be a string or numpy array, got {type(s)}")
    
    try:
        # Remove the 'array(' prefix and ')' suffix
        s = s.strip()
        if s.startswith('array(') and s.endswith(')'):
            s = s[6:-1]
        
        # Split the string into individual number strings
        num_strings = re.findall(r'-?\d+\.?\d*e?[-+]?\d*', s)
        
        # Convert to float and create a numpy array
        return np.array([float(num) for num in num_strings])
    except Exception as e:
        print(f"Error parsing string: {s}")
        raise e

def decode_array(obj):
    """
    Decode a JSON string to a NumPy array.

    Parameters
    ----------
    obj : str
        JSON string to decode.

    Returns
    -------
    np.array or original object
        Decoded NumPy array or original object if decoding fails.
    """
    if isinstance(obj, str):
        try:
            loaded = json.loads(obj)
            if isinstance(loaded, dict):
                if '__ndarray__' in loaded:
                    # Decode regular array
                    data = base64.b64decode(loaded['__ndarray__'])
                    return np.frombuffer(data, dtype=np.dtype(loaded['dtype'])).reshape(loaded['shape'])
                elif '__ndarray_of_arrays__' in loaded:
                    # Decode array of arrays
                    return np.array([decode_array(a) for a in loaded['__ndarray_of_arrays__']], dtype=object)
        except:
            pass
    return obj