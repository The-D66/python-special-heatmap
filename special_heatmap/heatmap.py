import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection

class SHeatmap:
    def __init__(self, data, fmt='sq', parent=None, cmap=None, sdata=None):
        self.data = np.array(data)
        self.fmt = fmt
        self.ax = parent
        self.sdata = np.array(sdata) if sdata is not None else None
        self.text_handles = {} # To store text objects for show_stars and set_text_mn
        self.patch_handles = {} # To store patch objects for set_patch and set_patch_mn
        self.box_handle = None # To store the box/frame
        
        # Default Colors (copied from SHeatmap.m)
        # Sequential (Greens)
        self.dfColor1 = np.array([
            [0.9686, 0.9882, 0.9412], [0.9454, 0.9791, 0.9199], [0.9221, 0.9700, 0.8987], [0.8988, 0.9609, 0.8774],
            [0.8759, 0.9519, 0.8560], [0.8557, 0.9438, 0.8338], [0.8354, 0.9357, 0.8115], [0.8152, 0.9276, 0.7892],
            [0.7909, 0.9180, 0.7685], [0.7545, 0.9039, 0.7523], [0.7180, 0.8897, 0.7361], [0.6816, 0.8755, 0.7199],
            [0.6417, 0.8602, 0.7155], [0.5962, 0.8430, 0.7307], [0.5507, 0.8258, 0.7459], [0.5051, 0.8086, 0.7610],
            [0.4596, 0.7873, 0.7762], [0.4140, 0.7620, 0.7914], [0.3685, 0.7367, 0.8066], [0.3230, 0.7114, 0.8218],
            [0.2837, 0.6773, 0.8142], [0.2483, 0.6378, 0.7929], [0.2129, 0.5984, 0.7717], [0.1775, 0.5589, 0.7504],
            [0.1421, 0.5217, 0.7314], [0.1066, 0.4853, 0.7132], [0.0712, 0.4488, 0.6950], [0.0358, 0.4124, 0.6768],
            [0.0314, 0.3724, 0.6364], [0.0314, 0.3319, 0.5929], [0.0314, 0.2915, 0.5494], [0.0314, 0.2510, 0.5059]
        ])
        
        # Diverging (Red-Blue)
        self.dfColor2 = np.array([
            [0.6196, 0.0039, 0.2588], [0.6892, 0.0811, 0.2753], [0.7588, 0.1583, 0.2917], [0.8283, 0.2354, 0.3082],
            [0.8706, 0.2966, 0.2961], [0.9098, 0.3561, 0.2810], [0.9490, 0.4156, 0.2658], [0.9660, 0.4932, 0.2931],
            [0.9774, 0.5755, 0.3311], [0.9887, 0.6577, 0.3690], [0.9930, 0.7266, 0.4176], [0.9943, 0.7899, 0.4707],
            [0.9956, 0.8531, 0.5238], [0.9968, 0.9020, 0.5846], [0.9981, 0.9412, 0.6503], [0.9994, 0.9804, 0.7161],
            [0.9842, 0.9937, 0.7244], [0.9526, 0.9810, 0.6750], [0.9209, 0.9684, 0.6257], [0.8721, 0.9486, 0.6022],
            [0.7975, 0.9183, 0.6173], [0.7228, 0.8879, 0.6325], [0.6444, 0.8564, 0.6435], [0.5571, 0.8223, 0.6448],
            [0.4698, 0.7881, 0.6460], [0.3868, 0.7461, 0.6531], [0.3211, 0.6727, 0.6835], [0.2553, 0.5994, 0.7139],
            [0.2016, 0.5261, 0.7378], [0.2573, 0.4540, 0.7036], [0.3130, 0.3819, 0.6694], [0.3686, 0.3098, 0.6353]
        ])

        # Determine Colormap and Range
        self.max_v = np.nanmax(np.abs(self.data))
        if np.any(self.data < 0):
            # Diverging
            self.colormap_data = self.dfColor2
            self.vmin, self.vmax = -self.max_v, self.max_v
            self.cmap_name = 'SHeatmap_Diverging'
        else:
            self.colormap_data = self.dfColor1[::-1] 
            self.vmin, self.vmax = 0, self.max_v
            self.cmap_name = 'SHeatmap_Sequential'

        if cmap is None:
            self.cmap = mcolors.LinearSegmentedColormap.from_list(self.cmap_name, self.colormap_data)
        else:
            self.cmap = plt.get_cmap(cmap)
        
        self.type = 'full'
            
        self.patches_list = []
        self.texts = []
        
    def set_type(self, type_str):
        """
        Set heatmap type: 'full', 'triu', 'tril', 'triu0', 'tril0'
        """
        self.type = type_str

    def draw(self, colorbar=True, colorbar_loc='right', colorbar_size="5%", colorbar_pad=0.1, mark_extremes=True):
        if self.ax is None:
            fig, self.ax = plt.subplots(figsize=(8, 8))
        
        rows, cols = self.data.shape
        
        # Setup Axes
        self.ax.set_xlim(0, cols)
        self.ax.set_ylim(rows, 0) # Invert Y axis (0 at top)
        self.ax.set_aspect('equal')
        self.ax.set_xticks(np.arange(cols) + 0.5)
        self.ax.set_yticks(np.arange(rows) + 0.5)
        self.ax.set_xticklabels(np.arange(1, cols + 1))
        self.ax.set_yticklabels(np.arange(1, rows + 1))
        self.ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        
        norm = mcolors.Normalize(vmin=self.vmin, vmax=self.vmax)
        
        for r in range(rows):
            for c in range(cols):
                # Visibility Logic
                visible = True
                if self.type == 'triu':
                    if c < r: visible = False
                elif self.type == 'tril':
                    if c > r: visible = False
                elif self.type == 'triu0':
                    if c <= r: visible = False
                elif self.type == 'tril0':
                    if c >= r: visible = False
                
                if not visible:
                    continue

                val = self.data[r, c]
                # Center coordinates
                cx, cy = c + 0.5, r + 0.5
                
                if np.isnan(val):
                    rect = patches.Rectangle((c, r), 1, 1, facecolor='0.8', edgecolor='none')
                    self.ax.add_patch(rect)
                    self.ax.text(cx, cy, '×', ha='center', va='center', fontsize=16, fontname='Times New Roman')
                    continue
                
                color = self.cmap(norm(val))
                t_ratio = abs(val) / self.max_v
                
                patch = None
                
                if self.fmt == 'sq':
                    patch = patches.Rectangle((c + 0.01, r + 0.01), 0.98, 0.98, facecolor=color, edgecolor='none')
                    self.ax.add_patch(patch)
                    
                elif self.fmt == 'asq':
                    size = 0.98 * t_ratio
                    offset = (1 - size) / 2
                    patch = patches.Rectangle((c + offset, r + offset), size, size, facecolor=color, edgecolor='none')
                    self.ax.add_patch(patch)
                    
                elif self.fmt == 'circ':
                    radius = 0.92 * 0.5
                    patch = patches.Circle((cx, cy), radius, facecolor=color, edgecolor='none')
                    self.ax.add_patch(patch)
                    
                elif self.fmt == 'acirc':
                    radius = 0.92 * 0.5 * t_ratio
                    patch = patches.Circle((cx, cy), radius, facecolor=color, edgecolor='none')
                    self.ax.add_patch(patch)
                
                elif self.fmt == 'bcirc':
                    bg_radius = 0.92 * 0.5
                    bg_circ = patches.Circle((cx, cy), bg_radius, facecolor='white', edgecolor='0.3', linewidth=0.8)
                    self.ax.add_patch(bg_circ)
                    radius = 0.92 * 0.5 * t_ratio
                    patch = patches.Circle((cx, cy), radius, facecolor=color, edgecolor='none')
                    self.ax.add_patch(patch)

                elif self.fmt == 'pie':
                    bg_radius = 0.92 * 0.5
                    bg_circ = patches.Circle((cx, cy), bg_radius, facecolor='white', edgecolor='0.3', linewidth=0.8)
                    self.ax.add_patch(bg_circ)
                    theta1 = 90
                    theta2 = 90 + (val / self.max_v) * 360
                    wedge = patches.Wedge((cx, cy), bg_radius, theta1, theta2, facecolor=color, edgecolor='0.3', linewidth=0.8)
                    self.ax.add_patch(wedge)

                elif self.fmt == 'donut':
                    # Donut chart is essentially a wedge with a width
                    bg_radius = 0.92 * 0.5
                    width = bg_radius * 0.5
                    bg_wedge = patches.Wedge((cx, cy), bg_radius, 0, 360, width=width, facecolor='white', edgecolor='0.3', linewidth=0.8)
                    self.ax.add_patch(bg_wedge)
                    theta1 = 90
                    theta2 = 90 + (val / self.max_v) * 360
                    wedge = patches.Wedge((cx, cy), bg_radius, theta1, theta2, width=width, facecolor=color, edgecolor='0.3', linewidth=0.8)
                    self.ax.add_patch(wedge)

                elif self.fmt == 'hex':
                    radius = 0.5 * 0.98 * t_ratio
                    poly = patches.RegularPolygon((cx, cy), numVertices=6, radius=radius, 
                                                  orientation=0, facecolor=color, edgecolor='0.3', linewidth=0.8)
                    self.ax.add_patch(poly)

                elif self.fmt == 'oval':
                    t_val = val / self.max_v
                    base_a = 1 + t_val if t_val <= 0 else 1
                    base_b = 1 - t_val if t_val >= 0 else 1
                    ellipse = patches.Ellipse((cx, cy), width=base_a*0.9, height=base_b*0.9, angle=45,
                                              facecolor=color, edgecolor='0.3', linewidth=0.8)
                    self.ax.add_patch(ellipse)

                elif self.fmt == 'star':
                    tValue = val / self.max_v
                    # MATLAB: linspace(0,2*pi,11)+pi/10. Alternating radius.
                    angles = np.linspace(0, 2*np.pi, 11) + np.pi/10
                    radii = np.ones(11) * 0.92 * 0.5 * tValue
                    radii[1::2] = radii[1::2] * 0.5
                    x = radii * np.cos(angles) + cx
                    y = radii * np.sin(angles) + cy
                    poly = patches.Polygon(np.column_stack([x, y]), closed=True, facecolor=color, edgecolor='0.3', linewidth=0.8)
                    self.ax.add_patch(poly)

                elif self.fmt in ['tril', 'trill']:
                    poly = patches.Polygon(np.array([[-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5]]) + [cx, cy], 
                                           closed=True, facecolor=color, edgecolor='none', linewidth=0.8)
                    self.ax.add_patch(poly)
                    
                elif self.fmt in ['triu', 'triur']:
                    poly = patches.Polygon(np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]]) + [cx, cy], 
                                           closed=True, facecolor=color, edgecolor='none', linewidth=0.8)
                    self.ax.add_patch(poly)

                elif self.fmt == 'triul':
                    poly = patches.Polygon(np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]]) + [cx, cy], 
                                           closed=True, facecolor=color, edgecolor='none', linewidth=0.8)
                    self.ax.add_patch(poly)

                elif self.fmt == 'trilr':
                    poly = patches.Polygon(np.array([[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]) + [cx, cy], 
                                           closed=True, facecolor=color, edgecolor='none', linewidth=0.8)
                    self.ax.add_patch(poly)

                elif self.fmt == 'cust' and self.sdata is not None:
                    # sdata assumed to be shape (2, N) or (N, 2). Let's accept (N, 2)
                    if self.sdata.shape[0] == 2:
                        sdata_xy = self.sdata.T
                    else:
                        sdata_xy = self.sdata
                    # MATLAB inverts Y: -obj.SData(2,:)
                    sdata_xy_adj = sdata_xy.copy()
                    sdata_xy_adj[:, 1] = -sdata_xy_adj[:, 1]
                    poly = patches.Polygon(sdata_xy_adj + [cx, cy], closed=True, facecolor=color, edgecolor='0.3', linewidth=0.8)
                    self.ax.add_patch(poly)

                elif self.fmt == 'acust' and self.sdata is not None:
                    if self.sdata.shape[0] == 2:
                        sdata_xy = self.sdata.T
                    else:
                        sdata_xy = self.sdata
                    sdata_xy_adj = sdata_xy.copy()
                    sdata_xy_adj[:, 1] = -sdata_xy_adj[:, 1]
                    poly = patches.Polygon((sdata_xy_adj * t_ratio) + [cx, cy], closed=True, facecolor=color, edgecolor='0.3', linewidth=0.8)
                    self.ax.add_patch(poly)

        # Colorbar
        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(self.ax)
            
            if colorbar_loc in ['right', 'left']:
                orientation = 'vertical'
            else:
                orientation = 'horizontal'
            
            cax = divider.append_axes(colorbar_loc, size=colorbar_size, pad=colorbar_pad)
            
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax, orientation=orientation)
            
            if mark_extremes:
                current_ticks = cbar.get_ticks()
                data_range = self.vmax - self.vmin
                threshold = data_range * 0.05
                
                final_ticks = [self.vmin, self.vmax]
                for t in current_ticks:
                    if t < self.vmin - 1e-9 or t > self.vmax + 1e-9:
                        continue
                    if abs(t - self.vmin) > threshold and abs(t - self.vmax) > threshold:
                        final_ticks.append(t)
                
                final_ticks = sorted(list(set(final_ticks)))
                cbar.set_ticks(final_ticks)
                
                def format_tick(x):
                    if abs(x) < 1e-10: return "0"
                    if abs(x) >= 0.01: return f"{x:.2f}"
                    return f"{x:.2e}"
                
                cbar.set_ticklabels([format_tick(x) for x in final_ticks])
        
        return self

    def set_text(self, fmt='{:.2f}', **kwargs):
        rows, cols = self.data.shape
        for r in range(rows):
            for c in range(cols):
                visible = True
                if self.type == 'triu':
                    if c < r: visible = False
                elif self.type == 'tril':
                    if c > r: visible = False
                elif self.type == 'triu0':
                    if c <= r: visible = False
                elif self.type == 'tril0':
                    if c >= r: visible = False
                
                if not visible:
                    continue

                val = self.data[r, c]
                if np.isnan(val):
                    continue
                
                s = fmt.format(val)
                cx, cy = c + 0.5, r + 0.5
                text_obj = self.ax.text(cx, cy, s, ha='center', va='center', **kwargs)
                self.text_handles[(r, c)] = text_obj

    def show_stars(self, pval, levels=None, corr_label=True):
        if levels is None:
            levels = [0.05, 0.01, 0.001]
        
        rows, cols = self.data.shape
        for r in range(rows):
            for c in range(cols):
                if np.isnan(self.data[r, c]):
                    continue
                
                p = pval[r, c]
                num_stars = sum(p < lvl for lvl in levels)
                stars_str = '*' * num_stars
                
                if (r, c) in self.text_handles:
                    text_obj = self.text_handles[(r, c)]
                    if corr_label:
                        old_str = text_obj.get_text()
                        # Use newline to simulate MATLAB's cell array multi-line text
                        new_str = f"{stars_str}\n{old_str}" if stars_str else old_str
                        text_obj.set_text(new_str)
                    else:
                        text_obj.set_text(stars_str)

if __name__ == '__main__':
    # Simple self-test
    data = np.random.rand(10, 10) - 0.5
    data[2, 2] = np.nan
    shm = SHeatmap(data, fmt='sq')
    shm.draw()
    plt.show()
