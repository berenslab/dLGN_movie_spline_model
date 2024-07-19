import ast
import numpy as np
import pandas as pd
import re
import scipy
import seaborn as sns
import base64
import json

def get_color_dict(palette_name='colorblind', n=None, color_names=None,
                   add_default_colors=True):
    """
    Returns color map dictionary from seaborn color palettes with hexadecimal triplet values.

    Parameters
    ----------
    palette_name : str
        Name of palette. Possible values are:
        'colorblind', 'deep', 'muted', 'bright', 'pastel', 'dark'
        For more options see seaborn.color_palette() docs.
    n : int
        Number of colors.
        If not None and arg color_names is given, arg n must be equal to len(color_names).
    color_names : list
        List of color names as strings e.g. for the color palette 'colorblind': ['blue',
        'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'yellow', 'lightblue']
        Must have the same length as colors in the color palette.
        If None, color names are set to ['c0', 'c1', 'c2', ...].
    add_default_colors : bool
        Add default colors that are need but that are not part of the color palette such as
        e.g. 'black'.
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

def get_formatted_pval(p, stars=False, show_ns_val=True):
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

def preprocess_array_string(s):
    """Define the preprocessing function for the .csv file
    """
    # Remove any newline characters
    s = s.replace('\n', ' ')
    # Ensure commas between numbers using a more robust regular expression
    s = re.sub(r'(?<=\d)\s+(?=-?\d)', ', ', s)
    return s

def str_to_array(s):
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
    if isinstance(obj, str):
        try:
            loaded = json.loads(obj)
            if isinstance(loaded, dict) and '__ndarray__' in loaded:
                data = base64.b64decode(loaded['__ndarray__'])
                return np.frombuffer(data, dtype=np.dtype(loaded['dtype'])).reshape(loaded['shape'])
        except:
            pass
    return obj