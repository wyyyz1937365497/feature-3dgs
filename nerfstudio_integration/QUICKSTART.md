# feature-3dgs 集成快速开始指南

本指南帮助您快速开始使用 feature-3dgs 与 nerfstudio 的集成。

## 目录结构

```
G:/TJ/feature-3dgs/
├── feature_3dgs/                  # 主模块
│   ├── __init__.py
│   ├── feature_3dgs.py           # 核心模型
│   ├── feature_3dgs_configs.py    # 方法配置
│   ├── dataparsers/
│   │   └── semantic_feature_dataparser.py
│   └── datasets/
│       └── semantic_feature_dataset.py
├── scripts/                       # 实用脚本
│   ├── precompute_semantic_features.py
│   ├── test_integration.py
│   └── editing_demo.py
├── third_party/
│   └── nerfstudio/                # Git 子模块（参考）
├── pyproject.toml                 # 包配置
├── INSTALL.md                      # 安装指南
└── README.md                       # 项目说明
```

## 安装步骤

### 1. 安装依赖

```bash
# 安装 nerfstudio
pip install nerfstudio

# 安装 gsplat
pip install gsplat>=1.0.0
```

### 2. 安装 feature-3dgs

在项目根目录下运行：

```bash
pip install -e .
```

### 3. 验证安装

```bash
ns-train --help
```

应该能看到 `feature-3dgs` 和 `feature-3dgs-speedup` 选项。

## 快速使用

### 预计算语义特征

```bash
python scripts/precompute_semantic_features.py \
    --data data/DATASET_NAME \
    --output data/DATASET_NAME/features \
    --model lseg \
    --resize 480 640
```

### 训练

```bash
# 标准训练
ns-train feature-3dgs --data data/DATASET_NAME

# 加速模式
ns-train feature-3dgs-speedup --data data/DATASET_NAME
```

### 文本编辑

```bash
python scripts/editing_demo.py \
    --checkpoint outputs/feature_3dgs_model/ \
    --text "chair" \
    --operation deletion
```

## 配置选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `semantic_feature_dim` | 512 | 语义特征维度 |
| `semantic_loss_weight` | 1.0 | 语义损失权重 |
| `use_speedup` | False | 启用 CNN 解码器加速 |
| `enable_editing` | True | 启用文本引导编辑 |

## 更多信息

- [完整安装指南](INSTALL.md)
- [实现总结](nerfstudio_integration/IMPLEMENTATION_SUMMARY.md)
- [文件清单](nerfstudio_integration/FILES_CREATED.md)
