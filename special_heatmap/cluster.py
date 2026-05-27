import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.cluster.hierarchy import linkage, dendrogram

def SDendrogram(data, orientation='top', method='average', ax=None, **kwargs):
    """
    Draw a dendrogram on the specified axes.
    """
    if ax is None:
        ax = plt.gca()
        
    if orientation in ['top', 'bottom']:
        Z = linkage(data.T, method=method)
    else:
        Z = linkage(data, method=method)
        
    out = dendrogram(
        Z, 
        orientation=orientation, 
        ax=ax, 
        link_color_func=lambda k: 'black', 
        **kwargs
    )
    
    for coll in ax.collections:
        coll.set_linewidth(0.8)
        
    ax.axis('off')
    
    # Return the order of the leaves
    return out['leaves']

def SClusterBlock(class_labels, orientation='top', min_lim=0, ax=None, color_list=None, block_prop=None):
    """
    Draw a cluster block on the specified axes to indicate grouping.
    """
    if ax is None:
        ax = plt.gca()
        
    class_labels = np.asarray(class_labels).flatten()
    
    if color_list is None:
        color_list = np.array([
            [0.5529, 0.8275, 0.7804], [1.0000, 1.0000, 0.7020], [0.7451, 0.7294, 0.8549],
            [0.9843, 0.5020, 0.4471], [0.5020, 0.6941, 0.8275], [0.9922, 0.7059, 0.3843],
            [0.7020, 0.8706, 0.4118], [0.9882, 0.8039, 0.8980], [0.8510, 0.8510, 0.8510],
            [0.7373, 0.5020, 0.7412], [0.8000, 0.9216, 0.7725], [1.0000, 0.9294, 0.4353]
        ])
        
    if block_prop is None:
        block_prop = {'linewidth': 0.8, 'edgecolor': 'none'}
        
    ax.axis('off')
    
    diffs = np.diff(class_labels)
    breaks = np.where(diffs != 0)[0]
    cc_list = np.concatenate(([0], breaks + 1, [len(class_labels)]))
    
    x_centers = []
    y_centers = []
    
    for i in range(len(cc_list) - 1):
        start_idx = cc_list[i]
        end_idx = cc_list[i+1]
        val = class_labels[start_idx]
        
        # Color index (1-based in MATLAB, so val-1)
        c_idx = int(val - 1) % len(color_list) if np.issubdtype(type(val), np.number) else 0
        color = color_list[c_idx]
        
        if orientation in ['top', 'bottom']:
            rect = patches.Rectangle((start_idx, min_lim), end_idx - start_idx, 1, facecolor=color, **block_prop)
            ax.add_patch(rect)
            x_centers.append((start_idx + end_idx) / 2.0)
            y_centers.append(min_lim + 0.5)
        else:
            rect = patches.Rectangle((min_lim, start_idx), 1, end_idx - start_idx, facecolor=color, **block_prop)
            ax.add_patch(rect)
            x_centers.append(min_lim + 0.5)
            y_centers.append((start_idx + end_idx) / 2.0)
            
    if orientation in ['top', 'bottom']:
        ax.set_xlim(0, len(class_labels))
        ax.set_ylim(min_lim, min_lim + 1)
    else:
        ax.set_ylim(len(class_labels), 0) # Invert Y to match heatmap
        ax.set_xlim(min_lim, min_lim + 1)
        
    return x_centers, y_centers
