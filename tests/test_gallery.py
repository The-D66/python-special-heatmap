import numpy as np
import matplotlib.pyplot as plt
import os
import pytest
from special_heatmap import SHeatmap

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
