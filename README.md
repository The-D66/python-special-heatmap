# Python Special Heatmap

[![PyPI version](https://badge.fury.io/py/special-heatmap.svg)](https://badge.fury.io/py/special-heatmap)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

> **Disclaimer**: This project was implemented with the assistance of AI (Google Gemini).  
> **声明**：本项目代码由 AI (Google Gemini) 辅助生成。

#### 介绍 / Introduction
A Python reproduction of the MATLAB [special heatmap](https://github.com/slandarer/matlab-special-heatmap) project. It uses `matplotlib` to draw advanced heatmaps with various shapes (square, circle, hexagon, pie, etc.) and triangular layouts.

基于 MATLAB 版 [special heatmap](https://github.com/slandarer/matlab-special-heatmap) 的 Python 复现。本项目使用 `matplotlib` 实现具有多种形状（正方形、圆形、六边形、饼图等）和三角矩阵布局的高级热图绘制。

### 1. 安装 / Installation

```bash
pip install special-heatmap
```

### 2. 基础使用 / Basic Use

#### 2.1 绘制非负矩阵热图 (Draw positive heat map)
```python
import numpy as np
import matplotlib.pyplot as plt
from special_heatmap import SHeatmap

data = np.random.rand(15, 15)
shm = SHeatmap(data, fmt='sq')
shm.draw()
plt.show()
```
![Basic Positive](gallery/Basic_positive.png)

#### 2.2 绘制含负数热图 (Contains negative numbers)
```python
data = np.random.rand(15, 15) - 0.5
shm = SHeatmap(data, fmt='sq')
shm.draw()
plt.show()
```
![Basic Negative](gallery/Basic_negative.png)

#### 2.3 绘制有文本和 NaN 的热图 (Draw heat map with texts and NaN)
```python
data = np.random.rand(12, 12) - 0.5
data[3, 3] = np.nan
shm = SHeatmap(data, fmt='sq')
shm.draw()
shm.set_text(fontsize=8)
plt.show()
```
![Basic with Text](gallery/Basic_with_text.png)

### 3. 各类型热图 / Various Formats

支持的格式 (`fmt` 参数):
- `sq` (Square): 正方形 (Default)
- `pie` (Pie Chart): 饼图
- `donut` (Donut Chart): 环形饼图(甜甜圈图)
- `circ` (Circle): 圆形
- `bcirc` (Circle with box): 有底色的圆形
- `oval` (Oval): 椭圆形
- `hex` (Hexagon): 六边形
- `star` (Star): 五角星
- `tril` / `trill`: 下三角形状
- `triu` / `triur`: 上三角形状
- `trilr`: 右下三角形状
- `triul`: 左上三角形状
- `asq` (Auto-Square): 自适应大小的正方形
- `acirc` (Auto-Circle): 自适应大小的圆形
- `cust` / `acust`: 自定义形状（配合 `sdata` 参数传入多边形顶点坐标）

```python
# 示例：绘制星形格式
shm = SHeatmap(data, fmt='star')
shm.draw()
```

| Format | Positive Data (A) | Mixed Data (B) |
| :---: | :---: | :---: |
| **sq** (Square) | ![sq A](gallery/Format_sq_A.png) | ![sq B](gallery/Format_sq_B.png) |
| **pie** (Pie Chart) | ![pie A](gallery/Format_pie_A.png) | ![pie B](gallery/Format_pie_B.png) |
| **donut** (Donut) | ![donut A](gallery/Format_donut_A.png) | ![donut B](gallery/Format_donut_B.png) |
| **circ** (Circle) | ![circ A](gallery/Format_circ_A.png) | ![circ B](gallery/Format_circ_B.png) |
| **bcirc** (Circle w/ Box) | ![bcirc A](gallery/Format_bcirc_A.png) | ![bcirc B](gallery/Format_bcirc_B.png) |
| **hex** (Hexagon) | ![hex A](gallery/Format_hex_A.png) | ![hex B](gallery/Format_hex_B.png) |
| **oval** (Oval) | ![oval A](gallery/Format_oval_A.png) | ![oval B](gallery/Format_oval_B.png) |
| **star** (Star) | ![star A](gallery/Format_star_A.png) | ![star B](gallery/Format_star_B.png) |
| **tril** (Lower Triangle) | ![tril A](gallery/Format_tril_A.png) | ![tril B](gallery/Format_tril_B.png) |
| **triu** (Upper Triangle) | ![triu A](gallery/Format_triu_A.png) | ![triu B](gallery/Format_triu_B.png) |
| **trilr** (Lower-Right Tri) | ![trilr A](gallery/Format_trilr_A.png) | ![trilr B](gallery/Format_trilr_B.png) |
| **triul** (Upper-Left Tri) | ![triul A](gallery/Format_triul_A.png) | ![triul B](gallery/Format_triul_B.png) |
| **asq** (Auto-Square) | ![asq A](gallery/Format_asq_A.png) | ![asq B](gallery/Format_asq_B.png) |
| **acirc** (Auto-Circle) | ![acirc A](gallery/Format_acirc_A.png) | ![acirc B](gallery/Format_acirc_B.png) |

自定义形状示例 (`cust` / `acust`):
| **cust** (Custom) | **acust** (Auto-Custom) |
| :---: | :---: |
| ![cust](gallery/Format_cust.png) | ![acust](gallery/Format_acust.png) |

### 4. 显示显著性 (Significance Stars)

可以使用 `show_stars` 方法根据 p-value 矩阵在热图上标注显著性星号。

```python
pval_matrix = np.random.rand(12, 12) * 0.1
shm = SHeatmap(data, fmt='sq')
shm.draw()
shm.set_text()
# 默认 levels 为 [0.05, 0.01, 0.001]
shm.show_stars(pval_matrix)
plt.show()
```
![Significance Stars](gallery/Significance_stars.png)

### 5. 三角布局 / Triangular Layouts

支持以下布局类型 (`set_type` 方法):
- `triu`   : 上三角 (Upper Triangle)
- `tril`   : 下三角 (Lower Triangle)
- `triu0`  : 上三角无对角线 (Upper without Diagonal)
- `tril0`  : 下三角无对角线 (Lower without Diagonal)

```python
data = np.random.rand(12, 12)
shm = SHeatmap(data, fmt='sq')
shm.set_type('tril') # 设置为下三角
shm.draw()
shm.set_text()
plt.show()
```

| Type | Result |
| :---: | :---: |
| **triu** | ![triu](gallery/Type_triu.png) |
| **tril** | ![tril](gallery/Type_tril.png) |
| **triu0** | ![triu0](gallery/Type_triu0.png) |
| **tril0** | ![tril0](gallery/Type_tril0.png) |

### 5. 开源协议 / License

本项目采用 **GPL v2** 开源协议。详细内容请参阅 [LICENSE](LICENSE) 文件。
This project is licensed under the **GPL v2 License**. See the [LICENSE](LICENSE) file for details.

---
*Original MATLAB project by [slandarer](https://github.com/slandarer)*