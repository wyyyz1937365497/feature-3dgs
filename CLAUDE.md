# feature-3dgs 项目记录

## 项目概述

feature-3dgs 是一个将 2D 基础模型（LSeg/SAM）的语义特征蒸馏到 3D 高斯表示中的项目。当前目标是将其整合到 nerfstudio 框架中以获得更高效的训练 pipeline。

**创建日期**: 2024-03-05

---

## 当前任务

### 任务：将 feature-3dgs 整合到 nerfstudio

**目标**: 将原始 feature-3dgs 训练代码整合到 nerfstudio 框架，实现：
1. 更高效的训练 pipeline
2. 模块化的代码架构
3. 与 nerfstudio 生态的集成

**关键决策**（已确认）:
- 渲染方案: gsplat 原生 N-D 特征支持（`sh_degree=None` 模式）
- 特征提取: 预计算特征图方式（训练前提取并保存 .pt 文件）
- 方法命名: `feature-3dgs`
- 编辑功能: 完整保留文本编辑功能

---

## 实现进度

### ✅ Phase 1: 文件结构和设置

**更新后的目录结构**（2024-03-05 重组）:
```
G:/TJ/feature-3dgs/
├── feature_3dgs_extension/         # 集成代码（独立模块）
│   ├── models/
│   │   ├── feature_3dgs.py         # 核心模型
│   │   └── __init__.py
│   ├── data/
│   │   ├── dataparsers/
│   │   │   ├── semantic_feature_dataparser.py
│   │   │   └── __init__.py
│   │   ├── datasets/
│   │   │   ├── semantic_feature_dataset.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── configs/
│   │   ├── feature_3dgs_configs.py
│   │   └── __init__.py
│   └── __init__.py
├── third_party/
│   └── nerfstudio/                 # Git submodule（只读参考）
├── scripts/                         # 工具脚本
│   ├── precompute_semantic_features.py
│   ├── register_feature_3dgs.py
│   ├── test_integration.py
│   └── editing_demo.py
└── nerfstudio_integration/          # 文档和安装
    ├── README.md
    ├── QUICKSTART.md
    ├── IMPLEMENTATION_SUMMARY.md
    ├── FILES_CREATED.md
    └── setup.py
```

**Git 子模块设置**:
- `third_party/nerfstudio`: https://github.com/nerfstudio-project/nerfstudio.git
- 使用 `git submodule update --init --recursive` 初始化
- 使用 `git submodule update --remote` 更新

### ✅ Phase 2: 语义特征数据层

**SemanticFeatureDataparser** (`feature_3dgs_extension/data/dataparsers/semantic_feature_dataparser.py`):
- 继承 `NerfstudioDataParser`
- 支持从 .pt 文件加载预计算的语义特征
- 处理变长特征（统一 resize 或 interpolate）
- speedup 模式支持（特征维度压缩）

**SemanticFeatureDataset** (`feature_3dgs_extension/data/datasets/semantic_feature_dataset.py`):
- 继承 `ImageDataset`
- 在 `get_metadata()` 中加载语义特征
- 处理特征图大小匹配

### ✅ Phase 3: 核心 feature-3dgs 模型

**Feature3DGSModel** (`feature_3dgs_extension/models/feature_3dgs.py`):

核心功能：
```python
@dataclass
class Feature3DGSModelConfig(SplatfactoModelConfig):
    semantic_feature_dim: int = 512
    use_semantic_features: bool = True
    semantic_loss_weight: float = 1.0
    use_speedup: bool = False
    enable_editing: bool = True
    edit_score_threshold: float = 0.5
```

关键方法：
- `populate_modules()`: 初始化语义特征参数
- `get_outputs()`: 使用 gsplat 渲染 RGB + 语义特征
- `get_loss_dict()`: 计算语义特征 L1 损失
- `render_edit()`: 文本引导的 3D 场景编辑
- `_calculate_selection_score()`: 语义相似度计算

**CNNDecoder**:
- 1x1 卷积解码器
- 用于 speedup 模式的特征解压

### ✅ Phase 4: 配置和注册

**方法配置** (`feature_3dgs_extension/configs/feature_3dgs_configs.py`):
- `feature-3dgs`: 标准配置
- `feature-3dgs-speedup`: CNN 解码器加速配置

优化器配置：
```python
"semantic_features": {
    "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
    "scheduler": None,
}
```

**导入方式**:
```python
# 需要先将项目根目录添加到 PYTHONPATH
import sys
from pathlib import Path
sys.path.insert(0, str(Path("path/to/feature-3dgs")))

from feature_3dgs_extension.models.feature_3dgs import Feature3DGSModel
```

### ✅ Phase 5: 工具脚本

| 脚本 | 功能 |
|------|------|
| `precompute_semantic_features.py` | 使用 LSeg/SAM 预计算特征 |
| `register_feature_3dgs.py` | 自动注册到 nerfstudio |
| `test_integration.py` | 集成测试套件 |
| `editing_demo.py` | 文本编辑演示 |

---

## 技术实现细节

### 1. 语义特征渲染

使用 gsplat 的 N-D 特征原生支持：

```python
semantic_render, semantic_alpha, _ = rasterization(
    means=means_crop,
    quats=quats_crop,
    scales=torch.exp(scales_crop),
    opacities=torch.sigmoid(opacities_crop).squeeze(-1),
    colors=semantic_crop,  # [N, D] 形状
    viewmats=viewmat,
    Ks=K,
    width=W,
    height=H,
    sh_degree=None,  # 关键：None 表示 N-D 特征
    # ... 其他参数 ...
)
```

### 2. 编辑功能实现

```python
def render_edit(self, camera, text_feature, edit_dict):
    # 计算语义相似度
    scores = self._calculate_selection_score(
        semantic_features, text_feature, edit_dict
    )

    # 执行编辑操作
    if "deletion" in op_dict:
        opacities.masked_fill_(scores >= 0.5, 0)
    if "extraction" in op_dict:
        opacities.masked_fill_(scores <= 0.5, 0)
    if "color_func" in op_dict:
        shs[:, 0, :] = shs[:, 0, :] * (1 - scores) + \
                       color_func(shs[:, 0, :]) * scores
```

### 3. 特征加载

预计算的特征以 .pt 文件存储：
- 文件命名: `<image_name>.pt`
- 格式: torch.Tensor，形状 [H, W, D]
- 加载时自动匹配图像文件名

---

## 命令行使用

### 训练

```bash
# 注册方法（首次使用）
python scripts/register_feature_3dgs.py

# 标准训练
ns-train feature-3dgs --data path/to/dataset

# 加速模式训练
ns-train feature-3dgs-speedup --data path/to/dataset
```

### 特征提取

```bash
python scripts/precompute_semantic_features.py \
    --data path/to/dataset \
    --output path/to/features \
    --model lseg \
    --resize 480 640
```

### 文本编辑

```bash
python scripts/editing_demo.py \
    --checkpoint path/to/checkpoint \
    --text "chair" \
    --operation deletion \
    --output edited.png
```

---

## 参考文件（原项目）

| 文件 | 用途 |
|------|------|
| `scene/gaussian_model.py` | 语义特征属性定义 |
| `gaussian_renderer/__init__.py` | 编辑功能实现 |
| `models/networks.py` | CNN/MLP 解码器 |
| `models/semantic_dataloader.py` | 数据加载逻辑 |

---

## 已知问题和限制

1. **特征维度匹配**: 需要 GT 特征维度与模型配置一致
2. **显存占用**: 512-dim 特征 + 100K 高斯 ≈ 额外 512MB
3. **Speedup 模式**: CNN 解码器可能损失一些语义信息

---

## 下一步

### 待完成

1. **测试和验证**:
   - [ ] 在样本数据上运行完整训练流程
   - [ ] 验证语义特征渲染质量
   - [ ] 性能基准测试（原始 vs 集成版本）

2. **Bug 修复**:
   - [ ] 处理特征加载的边缘情况
   - [ ] 修复 gsplat 兼容性问题
   - [ ] 测试不同 GPU 架构

3. **增强功能**:
   - [ ] 支持在线特征提取（无需预计算）
   - [ ] 多 GPU 训练支持
   - [ ] 更多编辑操作

---

## 文件清单

详见 `nerfstudio_integration/FILES_CREATED.md`

---

**最后更新**: 2024-03-05
