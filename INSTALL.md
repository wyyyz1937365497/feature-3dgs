# feature-3dgs 安装指南

本指南说明如何将 feature-3dgs 注册到 nerfstudio 并开始训练。

## 安装

### 1. 安装依赖

```bash
# 确保 nerfstudio 已安装
pip install nerfstudio

# 安装 gsplat
pip install gsplat>=1.0.0
```

### 2. 安装 feature-3dgs

在项目根目录下运行：

```bash
pip install -e .
```

这个命令会：
- 将 feature-3dgs 安装为可编辑包
- 注册 `feature-3dgs` 和 `feature-3dgs-speedup` 方法到 nerfstudio
- 注册 `semantic-feature` 数据解析器到 nerfstudio

### 3. 验证安装

运行以下命令检查方法是否已注册：

```bash
ns-train --help
```

在输出中应该能看到：
```
feature-3dgs      3D Gaussian Splatting with semantic feature support...
feature-3dgs-speedup  Feature-3dgs with CNN decoder for faster training...
```

### 4. 训练

```bash
# 标准训练
ns-train feature-3dgs --data data/DATASET_NAME

# 加速模式
ns-train feature-3dgs-speedup --data data/DATASET_NAME
```

---

## 预计算语义特征

在训练之前，你需要预计算语义特征：

```bash
python scripts/precompute_semantic_features.py \
    --data data/DATASET_NAME \
    --output data/DATASET_NAME/features \
    --model lseg \
    --resize 480 640
```

---

## 文本引导编辑

训练完成后，使用文本引导编辑：

```bash
python scripts/editing_demo.py \
    --checkpoint outputs/feature_3dgs_model/nerfstudio_models/ \
    --text "chair" \
    --operation deletion \
    --output edited_output.png
```

---

## 测试集成

运行集成测试套件：

```bash
python scripts/test_integration.py
```

---

## 故障排除

### 方法未找到

如果 `ns-train --help` 中没有显示 feature-3dgs，尝试：

1. 确认已安装：`pip show feature-3dgs`
2. 重新安装：`pip install -e . --force-reinstall --no-deps`
3. 检查 nerfstudio 版本：`ns-train --version`

### 导入错误

如果遇到导入错误，确认项目根目录在 PYTHONPATH 中：

```bash
# Linux/Mac
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Windows
set PYTHONPATH=%PYTHONPATH%;%CD%
```

---

## 目录结构

```
feature-3dgs/
├── feature_3dgs/                    # 主模块
│   ├── __init__.py
│   ├── feature_3dgs.py            # 模型实现
│   ├── feature_3dgs_configs.py    # 方法配置
│   ├── dataparsers/
│   │   ├── __init__.py
│   │   └── semantic_feature_dataparser.py
│   └── datasets/
│       ├── __init__.py
│       └── semantic_feature_dataset.py
├── scripts/                        # 实用脚本
│   ├── precompute_semantic_features.py
│   ├── test_integration.py
│   └── editing_demo.py
├── third_party/
│   └── nerfstudio/                # Git 子模块（参考）
├── pyproject.toml                  # 包配置
├── INSTALL.md                      # 本文件
└── README.md                       # 项目说明
```

---

## 开发者信息

本项目遵循 [nerfstudio 方法模板](https://github.com/nerfstudio-project/nerfstudio-method-template) 的推荐结构。

### 注册方式

使用 Python entrypoints 在 `pyproject.toml` 中注册：

```toml
[project.entry-points.'nerfstudio.method_configs']
feature-3dgs = 'feature_3dgs.feature_3dgs_configs:feature_3dgs'
feature-3dgs-speedup = 'feature_3dgs.feature_3dgs_configs:feature_3dgs_speedup'
```

这符合 nerfstudio 官方推荐的最佳实践。
