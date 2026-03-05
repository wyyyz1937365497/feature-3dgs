# Nerfstudio 本地修改说明

由于 feature-3dgs 使用函数形式的 entry points 来避免循环导入，需要对本地安装的 nerfstudio 进行以下修改。

## 修改内容

### 1. `nerfstudio/plugins/registry.py`

在第 42-51 行，添加对 callable 的支持：

```python
discovered_entry_points = entry_points(group="nerfstudio.method_configs")
for name in discovered_entry_points.names:
    spec = discovered_entry_points[name].load()

    # 支持 callable（函数）入口点以避免循环导入
    if callable(spec):
        spec = spec()

    if not isinstance(spec, MethodSpecification):
        CONSOLE.print(
            f"[bold yellow]Warning: Could not entry point {spec} as it is not an instance of MethodSpecification"
        )
        continue
    spec = t.cast(MethodSpecification, spec)
    methods[spec.config.method_name] = spec.config
    descriptions[spec.config.method_name] = spec.description
```

### 2. `nerfstudio/plugins/registry_dataparser.py`

在第 57-67 行，添加对 callable 的支持：

```python
discovered_entry_points = entry_points(group="nerfstudio.dataparser_configs")
for name in discovered_entry_points.names:
    spec = discovered_entry_points[name].load()

    # 支持 callable（函数）入口点以避免循环导入
    if callable(spec):
        spec = spec()

    if not isinstance(spec, DataParserSpecification):
        CONSOLE.print(
            f"[bold yellow]Warning: Could not entry point {spec} as it is not an instance of DataParserSpecification"
        )
        continue
    spec = t.cast(DataParserSpecification, spec)
    dataparsers[name] = spec.config
    descriptions[name] = spec.description
```

## 应用修改

如果你是从 `third_party/nerfstudio` 安装的 nerfstudio，修改已经存在，无需额外操作。

如果你是使用 `pip install nerfstudio` 安装的，需要：

```bash
# 找到 nerfstudio 安装位置
python -c "import nerfstudio; print(nerfstudio.__file__)"

# 手动编辑上述两个文件
```

## 说明

这些修改允许 nerfstudio 在加载 entry points 时调用函数来获取配置对象，而不是直接要求配置对象作为模块级变量。这避免了循环导入问题，因为函数内部的导入是在函数被调用时才执行的。
