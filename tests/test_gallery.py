import numpy as np
import matplotlib.pyplot as plt
import os
import pytest
from special_heatmap import SHeatmap, SDendrogram, SClusterBlock

# Output directory for generated images
GALLERY_DIR = "gallery"


@pytest.fixture(scope="session", autouse=True)
def ensure_gallery_dir():
    if not os.path.exists(GALLERY_DIR):
        os.makedirs(GALLERY_DIR)

def test_basic_positive():
    """Test generating basic positive heatmap."""
    data = np.random.rand(15, 15)
    shm = SHeatmap(data, fmt='sq')
    shm.draw()
    # plt.title('Basic Positive')
    output_path = os.path.join(GALLERY_DIR, 'Basic_positive.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    assert os.path.exists(output_path)

def test_basic_negative():
    """Test generating basic negative heatmap."""
    data = np.random.rand(15, 15) - 0.5
    shm = SHeatmap(data, fmt='sq')
    shm.draw()
    # plt.title('Basic Negative')
    output_path = os.path.join(GALLERY_DIR, 'Basic_negative.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    assert os.path.exists(output_path)

@pytest.mark.parametrize("fmt", ['sq', 'pie', 'circ', 'oval', 'hex', 'asq', 'acirc', 'star', 'donut', 'bcirc', 'tril', 'triu', 'trilr', 'triul'])
def test_formats(fmt):
    """Test generating heatmaps for all supported formats."""
    data_a = np.random.rand(12, 12)
    data_b = np.random.rand(12, 12) - 0.5
    
    # A (Positive)
    plt.figure()
    shm = SHeatmap(data_a, fmt=fmt)
    shm.draw()
    path_a = os.path.join(GALLERY_DIR, f'Format_{fmt}_A.png')
    plt.savefig(path_a, bbox_inches='tight', dpi=150)
    plt.close()
    assert os.path.exists(path_a)
    
    # B (Negative/Mixed)
    plt.figure()
    shm = SHeatmap(data_b, fmt=fmt)
    shm.draw()
    path_b = os.path.join(GALLERY_DIR, f'Format_{fmt}_B.png')
    plt.savefig(path_b, bbox_inches='tight', dpi=150)
    plt.close()
    assert os.path.exists(path_b)

def test_custom_formats():
    """Test generating heatmaps with custom polygon shapes."""
    data = np.random.rand(8, 8)
    # Define a simple triangle as custom shape
    sdata = np.array([[-0.4, 0.4], [0.4, 0.4], [0.0, -0.4]])
    
    for fmt in ['cust', 'acust']:
        plt.figure()
        shm = SHeatmap(data, fmt=fmt, sdata=sdata)
        shm.draw()
        path = os.path.join(GALLERY_DIR, f'Format_{fmt}.png')
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        assert os.path.exists(path)

def test_show_stars():
    """Test the significance marking feature."""
    data = np.random.rand(8, 8)
    pval = np.random.rand(8, 8) * 0.1 # Generate p-values mostly between 0 and 0.1
    
    plt.figure()
    shm = SHeatmap(data, fmt='sq')
    shm.draw()
    shm.set_text()
    shm.show_stars(pval)
    
    output_path = os.path.join(GALLERY_DIR, 'Significance_stars.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    assert os.path.exists(output_path)

def test_text_nan():
    """Test generating heatmap with Text and NaN values."""
    data = np.random.rand(12, 12) - 0.5
    # Set some NaNs
    data[3, 3] = np.nan
    data[4, 4] = np.nan
    data[11, 11] = np.nan
    
    shm = SHeatmap(data, fmt='sq')
    shm.draw()
    shm.set_text(fontsize=8, color='black') 
    # plt.title('Basic with Text and NaN')
    output_path = os.path.join(GALLERY_DIR, 'Basic_with_text.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    assert os.path.exists(output_path)

@pytest.mark.parametrize("layout_type", ['triu', 'tril', 'triu0', 'tril0'])
def test_triangle_layouts(layout_type):
    """Test generating triangular layout heatmaps."""
    data = np.random.rand(12, 12)
    
    plt.figure()
    shm = SHeatmap(data, fmt='sq')
    shm.set_type(layout_type)
    shm.draw()
    shm.set_text(fontsize=6, color='black')
    # plt.title(f'Triangular: {layout_type}')
    output_path = os.path.join(GALLERY_DIR, f'Type_{layout_type}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    assert os.path.exists(output_path)

if __name__ == "__main__":
    # Allow running this script directly as well
    pytest.main([__file__])

def test_tree():
    """Test Heatmap with dendrograms."""
    data = np.random.rand(10, 12)
    fig = plt.figure(figsize=(8, 7))
    
    ax_main = fig.add_axes([0.18, 0.07, 0.62, 0.77])
    ax_tree_l = fig.add_axes([0.05, 0.07, 0.12, 0.77])
    ax_tree_t = fig.add_axes([0.18, 0.85, 0.62, 0.12])
    
    order_l = SDendrogram(data, orientation='left', ax=ax_tree_l)
    order_t = SDendrogram(data.T, orientation='top', ax=ax_tree_t)
    
    # Reorder data
    data_reordered = data[order_l, :][:, order_t]
    
    shm = SHeatmap(data_reordered, fmt='sq', parent=ax_main)
    shm.draw(colorbar=True, colorbar_loc='right')
    
    output_path = os.path.join(GALLERY_DIR, 'Tree.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    assert os.path.exists(output_path)

def test_group():
    """Test Heatmap with cluster blocks."""
    data = np.random.rand(10, 12)
    class_row = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
    class_col = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
    
    fig = plt.figure(figsize=(8, 8))
    
    ax_main = fig.add_axes([0.15, 0.1, 0.75, 0.75])
    ax_block_l = fig.add_axes([0.10, 0.1, 0.04, 0.75])
    ax_block_t = fig.add_axes([0.15, 0.86, 0.75, 0.04])
    
    SClusterBlock(class_row, orientation='left', ax=ax_block_l)
    SClusterBlock(class_col, orientation='top', ax=ax_block_t)
    
    shm = SHeatmap(data, fmt='sq', parent=ax_main)
    shm.draw(colorbar=False)
    
    output_path = os.path.join(GALLERY_DIR, 'Group.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    assert os.path.exists(output_path)

def test_multilayer():
    """Test Heatmap with multilayer grouping."""
    data = np.random.rand(3, 16)
    class1 = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    class2 = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    
    fig = plt.figure(figsize=(10, 3.2))
    
    ax_main = fig.add_axes([0.05, 0.1, 0.9, 0.6])
    ax_block_t = fig.add_axes([0.05, 0.72, 0.9, 0.2])
    
    # Since we plot two blocks in the same axes, we use different min_lim
    SClusterBlock(class1, orientation='top', min_lim=1, ax=ax_block_t)
    SClusterBlock(class2, orientation='top', min_lim=0, ax=ax_block_t)
    
    shm = SHeatmap(data, fmt='sq', parent=ax_main)
    shm.draw(colorbar_loc='bottom')
    
    output_path = os.path.join(GALLERY_DIR, 'Multilayer.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    assert os.path.exists(output_path)

def test_treegroup():
    """Test Heatmap with both dendrogram and cluster blocks."""
    data = np.random.rand(10, 12)
    fig = plt.figure(figsize=(8, 7))
    
    ax_main = fig.add_axes([0.22, 0.07, 0.58, 0.77])
    
    ax_tree_l = fig.add_axes([0.05, 0.07, 0.1, 0.77])
    ax_tree_t = fig.add_axes([0.22, 0.88, 0.58, 0.1])
    
    order_l = SDendrogram(data, orientation='left', ax=ax_tree_l)
    order_t = SDendrogram(data.T, orientation='top', ax=ax_tree_t)
    
    ax_block_l = fig.add_axes([0.16, 0.07, 0.05, 0.77])
    ax_block_t = fig.add_axes([0.22, 0.85, 0.58, 0.02])
    
    # Just mock cluster labels for simplicity in test
    class_row = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    class_col = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    
    SClusterBlock(np.array(class_row)[order_l], orientation='left', ax=ax_block_l)
    SClusterBlock(np.array(class_col)[order_t], orientation='top', ax=ax_block_t)
    
    data_reordered = data[order_l, :][:, order_t]
    
    shm = SHeatmap(data_reordered, fmt='sq', parent=ax_main)
    shm.draw(colorbar=True, colorbar_loc='right')
    
    output_path = os.path.join(GALLERY_DIR, 'TreeGroup.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    assert os.path.exists(output_path)

