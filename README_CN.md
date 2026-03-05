# Feature 3DGS 🪄: 通过蒸馏特征场增强 3D Gaussian Splatting


Shijie Zhou, Haoran Chang*, Sicheng Jiang*, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, Achuta Kadambi (*表示同等贡献)<br>
| [网页](https://feature-3dgs.github.io/) | [论文全文](https://arxiv.org/abs/2312.03203) | [视频](https://www.youtube.com/watch?v=h4zmQsCV_Qw) | [Windows 预编译查看器](https://drive.google.com/file/d/1DRFrtFUfz27QvQKOWbYXbRS2o2eSgaUT/view?usp=sharing)<br>
![展示图片](assets/teaser_v5_2.png)

摘要：*3D场景表示在近年来获得了极大的流行。使用神经辐射场的方法在传统任务（如新视角合成）方面具有通用性。最近，一些工作旨在扩展 NeRF 的功能，使其超越视图合成，通过从 2D 基础模型蒸馏 3D 特征场来实现语义感知任务，如编辑和分割。然而，这些方法有两个主要限制：(a) 它们受限于 NeRF 管道的渲染速度，(b) 隐式表示的特征场存在连续性伪影，降低了特征质量。最近，3D Gaussian Splatting 在实时辐射场渲染方面表现出了最先进的性能。在这项工作中，我们更进一步：除了辐射场渲染外，我们还通过 2D 基础模型蒸馏实现了任意维度语义特征的 3D Gaussian splatting。这种转换并不简单：天真地将特征场整合到 3DGS 框架中会遇到重大挑战，特别是 RGB 图像和特征图之间的空间分辨率和通道一致性的差异。我们提出了架构和训练变更来有效避免这个问题。我们提出的方法是通用的，我们的实验展示了新视角语义分割、语言引导的编辑以及通过从最先进的 2D 基础模型（如 SAM 和 CLIP-LSeg）学习特征场来实现"分割一切"。在实验中，我们的蒸馏方法能够提供相当或更好的结果，同时训练和渲染速度明显更快。此外，据我们所知，我们是第一个通过利用 SAM 模型实现点和边界框提示来操作辐射场的方法。*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">引用</h2>
    <pre><code>@inproceedings{zhou2024feature,
  title={Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields},
  author={Zhou, Shijie and Chang, Haoran and Jiang, Sicheng and Fan, Zhiwen and Zhu, Zehao and Xu, Dejia and Chari, Pradyumna and You, Suya and Wang, Zhangyang and Kadambi, Achuta},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21676--21685},
  year={2024}
}</code></pre>
  </div>
</section>


# 环境设置
我们默认提供的安装方法基于 Conda 包和环境管理：

```shell
conda create --name feature_3dgs python=3.8
conda activate feature_3dgs
```

PyTorch（请检查您的 CUDA 版本，我们使用的是 11.8）
```shell
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

必需的包
```shell
pip install -r requirements.txt
```

子模块

<span style="color: orange;">***新***</span>：我们的并行 N 维高斯光栅化器现在支持 RGB、任意 N 维特征和深度渲染。
```shell
pip install submodules/diff-gaussian-rasterization-feature # RGB、n 维特征、深度的光栅化器
pip install submodules/simple-knn
```

# 处理您自己的场景

我们遵循与 [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) 相同的数据集处理方式。如果您想使用自己的场景，将您想使用的图像放在目录 ```<location>/input``` 中。
```
<location>
|---input
    |---<图像 0>
    |---<图像 1>
    |---...
```
对于光栅化，相机模型必须是 SIMPLE_PINHOLE 或 PINHOLE 相机。我们提供了一个转换器脚本 ```convert.py```，用于从输入图像中提取未失真的图像和 SfM 信息。或者，您可以使用 ImageMagick 调整未失真图像的大小。这种缩放类似于 MipNeRF360，即在相应的文件夹中创建原始分辨率的 1/2、1/4 和 1/8 的图像。要使用它们，请首先安装最新版本的 COLMAP（理想情况下支持 CUDA）和 ImageMagick。

如果您在系统路径中有 COLMAP 和 ImageMagick，可以简单地运行
```shell
python convert.py -s <location> [--resize] #如果不缩放，不需要 ImageMagick
```

我们的 COLMAP 加载器期望源路径位置中具有以下数据集结构：

```
<location>
|---images
|   |---<图像 0>
|   |---<图像 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

或者，您可以使用可选参数 ```--colmap_executable``` 和 ```--magick_executable``` 来指向相应的路径。请注意，在 Windows 上，可执行文件应指向负责设置执行环境的 COLMAP ```.bat``` 文件。完成后，```<location>``` 将包含预期的 COLMAP 数据集结构，其中包含未失真的、调整大小的输入图像，以及您的原始图像和一些临时（失真）数据在目录 ```distorted``` 中。

如果您有自己的没有失真的 COLMAP 数据集（例如，使用 ```OPENCV``` 相机），可以尝试只运行脚本的最后一部分：将图像放在 ```input``` 中，将 COLMAP 信息放在子目录 ```distorted``` 中：
```
<location>
|---input
   |---<图像 0>
   |---<图像 1>
   |---...
|---distorted
    |---database.db
    |---sparse
        |---0
            |---...
```
然后运行
```shell
python convert.py -s <location> --skip_matching [--resize] #如果不缩放，不需要 ImageMagick
```

<details>
<summary><span style="font-weight: bold;">convert.py 命令行参数</span></summary>

  #### --no_gpu
  避免在 COLMAP 中使用 GPU 的标志。
  #### --skip_matching
  表示图像的 COLMAP 信息可用的标志。
  #### --source_path / -s
  输入的位置。
  #### --camera
  用于早期匹配步骤的相机模型，默认为 ```OPENCV```。
  #### --resize
  创建调整大小的输入图像版本的标志。
  #### --colmap_executable
  COLMAP 可执行文件的路径（Windows 上为 ```.bat```）。
  #### --magick_executable
  ImageMagick 可执行文件的路径。
</details>
<br>




# 教师网络的特征编码

## LSeg 编码器

下载 LSeg 模型文件 `demo_e200.ckpt` 从 [Google drive](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing)，并将其放在文件夹下：`encoders/lseg_encoder`。

### 特征嵌入
```
cd encoders/lseg_encoder
python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir ../../data/DATASET_NAME/rgb_feature_langseg --test-rgb-dir ../../data/DATASET_NAME/images --workers 0
```
这可能会在 `--outdir` 中产生大的特征图文件（每个文件 100-200MB）。

运行 train.py。如果重建失败，将 `--scale 4.0` 更改为更小或更大的值，例如 `--scale 1.0` 或 `--scale 16.0`。





## SAM 编码器

### 安装

代码需要 `python>=3.8`，以及 `pytorch>=1.7` 和 `torchvision>=0.8`。请按照[此处](https://pytorch.org/get-started/locally/)的说明安装两个 PyTorch 和 TorchVision 依赖项。强烈建议安装支持 CUDA 的 PyTorch 和 TorchVision。

SAM 设置：
```
cd encoders/sam_encoder
pip install -e .
```

预训练模型下载：

点击下面的链接下载相应模型类型的检查点。

- **`default` 或 `vit_h`: [ViT-H SAM 模型。](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM 模型。](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM 模型。](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

并将其放在文件夹下：```
encoders/sam_encoder/checkpoints``

### 特征嵌入
运行以下命令以导出输入图像或图像目录的图像嵌入。
```
cd encoders/sam_encoder
python export_image_embeddings.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --input ../../data/DATASET_NAME/images  --output ../../data/OUTPUT_NAME/sam_embeddings
```



# 训练、渲染和推理：

## 🔥 新功能：多功能交互式查看器（可选）
我们很高兴介绍一个全新的多功能交互式查看器，用于可视化 *RGB*、*深度*、*边缘*、*法线*、*曲率*，特别是 <span style="color: orange;">***语义特征***</span>。Windows 的预编译查看器位于 `viewer_windows` 中，也可以[在此处](https://drive.google.com/file/d/1DRFrtFUfz27QvQKOWbYXbRS2o2eSgaUT/view?usp=sharing)下载。如果您的操作系统是 Ubuntu 22.04，则需要在本地编译查看器：
```shell
# 依赖项
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
# 项目设置
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # 添加 -G Ninja 以加快构建速度
cmake --build build -j24 --target install
```

您可以访问 [GS Monitor](https://github.com/RongLiu-Leo/Gaussian-Splatting-Monitor) 了解更多详情。


https://github.com/RongLiu-Leo/feature-3dgs/assets/102014841/7baf236f-29bc-4de1-9a99-97d528f6e63e
### 使用方法
首先运行查看器，
```shell
./viewer_windows/bin/SIBR_remoteGaussian_app_rwdi # Windows
```
或

```shell
./<SIBR 安装目录>/bin/SIBR_remoteGaussian_app # Ubuntu 22.04
```

然后

1. 如果您想监控训练过程，请运行 `train.py`，有关更多详细信息，请参阅[训练](#训练)部分。

2. 如果您更喜欢快速训练，请在训练完成后运行 `view.py` 与训练好的模型进行交互。有关更多详细信息，请参阅[查看训练好的模型](#查看训练好的模型)部分。


## 训练
```
python train.py -s data/DATASET_NAME -m output/OUTPUT_NAME -f lseg --speedup --iterations 7000
```
<details>
<summary><span style="font-weight: bold;">train.py 命令行参数</span></summary>

  #### --source_path / -s
  包含 COLMAP 或合成 NeRF 数据集的源目录路径。
  #### --model_path / -m
  训练模型应存储的路径（默认为 ```output/<random>```）。
  #### --foundation_model / -f
  切换不同的基础模型编码器，`lseg` 表示 LSeg，`sam` 表示 SAM
  #### --images / -i
  COLMAP 图像的替代子目录（默认为 ```images```）。
  #### --eval
  添加此标志以使用 MipNeRF360 风格的训练/测试分割进行评估。
  #### --resolution / -r
  指定训练前加载的图像的分辨率。如果提供 ```1、2、4``` 或 ```8```，则分别使用原始、1/2、1/4 或 1/8 分辨率。如果提供 ```0```，则使用 GT 特征图的分辨率。对于所有其他值，将宽度重新缩放到给定数字，同时保持图像纵横比。如果提供 ```-2```，则使用自定义分辨率（```utils/camera_utils.py L31```）。**如果未设置且输入图像宽度超过 1.6K 像素，输入将自动重新缩放到此目标。**
  #### --speedup
  可选的加速模块，用于减少特征维度初始化。
  #### --data_device
  指定源图像数据的位置，默认为 ```cuda```，建议在大型/高分辨率数据集上训练时使用 ```cpu```，这将减少 VRAM 消耗，但会稍微减慢训练速度。感谢 [HrsPythonix](https://github.com/HrsPythonix)。
  #### --white_background / -w
  添加此标志以使用白色背景而不是黑色（默认），例如，用于评估 NeRF 合成数据集。
  #### --sh_degree
  要使用的球谐函数阶数（不大于 3）。默认为 ```3```。
  #### --convert_SHs_python
  标志使管道使用 PyTorch 而不是我们的来计算 SH 的前向和后向。
  #### --convert_cov3D_python
  标志使管道使用 PyTorch 而不是我们的来计算 3D 协方差的前向和后向。
  #### --debug
  如果遇到错误，则启用调试模式。如果光栅化器失败，则会创建一个 ```dump``` 文件，您可以在问题中将其转发给我们，以便我们查看。
  #### --debug_from
  调试**很慢**。您可以指定一个迭代（从 0 开始），之后上述调试变为活动状态。
  #### --iterations
  训练的总迭代次数，默认为 ```30_000```。
  #### --ip
  启动 GUI 服务器的 IP，默认为 ```127.0.0.1```。
  #### --port
  GUI 服务器使用的端口，默认为 ```6009```。
  #### --test_iterations
  训练脚本在测试集上计算 L1 和 PSNR 的空格分隔迭代，默认为 ```7000 30000```。
  #### --save_iterations
  训练脚本保存高斯模型的空格分隔迭代，默认为 ```7000 30000 <iterations>```。
  #### --checkpoint_iterations
  存储检查点以供继续使用的空格分隔迭代，保存在模型目录中。
  #### --start_checkpoint
  保存的检查点的路径，用于从中继续训练。
  #### --quiet
  标志以省略任何写入标准输出管道的文本。
  #### --feature_lr
  球谐特征学习率，默认为 ```0.0025```。
  #### --opacity_lr
  不透明度学习率，默认为 ```0.05```。
  #### --scaling_lr
  缩放学习率，默认为 ```0.005```。
  #### --rotation_lr
  旋转学习率，默认为 ```0.001```。
  #### --position_lr_max_steps
  位置学习率从 ```initial``` 到 ```final``` 的步数（从 0 开始）。默认为 ```30_000```。
  #### --position_lr_init
  初始 3D 位置学习率，默认为 ```0.00016```。
  #### --position_lr_final
  最终 3D 位置学习率，默认为 ```0.0000016```。
  #### --position_lr_delay_mult
  位置学习率乘数（参考 Plenoxels），默认为 ```0.01```。
  #### --densify_from_iter
  密集化开始的迭代，默认为 ```500```。
  #### --densify_until_iter
  密集化停止的迭代，默认为 ```15_000```。
  #### --densify_grad_threshold
  根据 2D 位置梯度决定是否应该密集化点的限制，默认为 ```0.0002```。
  #### --densification_interval
  密集化的频率，默认为 ```100```（每 100 次迭代）。
  #### --opacity_reset_interval
  重置不透明度的频率，默认为 ```3_000```。
  #### --lambda_dssim
  SSIM 对总损失的影响从 0 到 1，默认为 ```0.2```。
  #### --percent_dense
  点必须超过的场景范围百分比（0--1）才能被强制密集化，默认为 ```0.01```。

</details>

### 高维特征的高斯光栅化
您可以在 `submodules/diff-gaussian-rasterization-feature/cuda_rasterizer/config.h` 中自定义 `NUM_SEMANTIC_CHANNELS` 以获得您想要的任意数量的特征维度：

- 在 `config.h` 中自定义 `NUM_SEMANTIC_CHANNELS`。

如果您想使用可选的 CNN 加速模块，请相应执行以下操作：

- 在 `scene/gaussian_model.py` 第 142 行的自定义 `semantic_feature_size/NUMBER` 中的 `NUMBER`。
- 在 `train.py` 第 51 行的自定义 `feature_out_dim/NUMBER` 中的 `NUMBER`。
- 在 `render.py` 第 117 行和 261 行的自定义 `feature_out_dim/NUMBER` 中的 `NUMBER`。

其中 `feature_out_dim` / `NUMBER` = `NUM_SEMANTIC_CHANNELS`。`feature_out_dim` 与地面实况基础模型维度匹配，LSeg 为 512，SAM 为 256。默认 `NUMBER = 4`。供您参考，以下是运行 `train.py` 的 4 种配置：

对于语言引导的编辑：

`-f lseg` 配合 `NUM_SEMANTIC_CHANNELS` `512`*（此任务无加速）。

对于分割任务：

`-f lseg --speedup` 配合 `NUM_SEMANTIC_CHANNELS` `128`，`NUMBER = 4`*。

`-f sam` 配合 `NUM_SEMANTIC_CHANNELS` `256`。

`-f sam --speedup` 配合 `NUM_SEMANTIC_CHANNELS` `64`，`NUMBER = 4`*。

*：我们实验中使用的设置
#### 注意：
每次修改任何 CUDA 代码后，请务必删除 `submodules/diff-gaussian-rasterization-feature/build` 并重新编译：
```
pip install submodules/diff-gaussian-rasterization-feature
```

## 查看训练好的模型
训练完成后，您可以在保持查看器运行的同时直接查看训练好的模型：
```shell
python view.py -s <COLMAP 或 NeRF 合成数据集的路径> -m <训练好的模型的路径> -f lseg
```
<details>
<summary><span style="font-weight: bold;">view.py 的重要命令行参数</span></summary>

  #### --source_path / -s
  包含 COLMAP 或合成 NeRF 数据集的源目录路径。
  #### --model_path / -m
  训练模型应存储的路径（默认为 ```output/<random>```）。
  #### --iteration
  指定要加载的迭代。
  #### -f
  sam 或 lseg

</details>
<br>

## 渲染
1. 从训练和测试视图渲染：
```
python render.py -s data/DATASET_NAME -m output/OUTPUT_NAME  --iteration 3000
```
<details>
<summary><span style="font-weight: bold;">render.py 命令行参数</span></summary>

  #### --model_path / -m
  要为其创建渲染的训练模型目录的路径。
  #### --skip_train
  跳过渲染训练集的标志。
  #### --skip_test
  跳过渲染测试集的标志。
  #### --quiet
  跳过任何写入标准输出管道的文本的标志。

  **以下参数将根据训练期间使用的内容自动从模型路径读取。但是，您可以通过在命令行上显式提供它们来覆盖它们。**

  #### --source_path / -s
  包含 COLMAP 或合成 NeRF 数据集的源目录路径。
  #### --images / -i
  COLMAP 图像的替代子目录（默认为 ```images```）。
  #### --eval
  添加此标志以使用 MipNeRF360 风格的训练/测试分割进行评估。
  #### --resolution / -r
  更改训练前加载的图像的分辨率。如果提供 ```1、2、4``` 或 ```8```，则分别使用原始、1/2、1/4 或 1/8 分辨率。对于所有其他值，将宽度重新缩放到给定数字，同时保持图像纵横比。默认为 ```1```。
  #### --white_background / -w
  添加此标志以使用白色背景而不是黑色（默认），例如，用于评估 NeRF 合成数据集。
  #### --convert_SHs_python
  标志使管道使用从 PyTorch 计算的 SH 而不是我们的来渲染。
  #### --convert_cov3D_python
  标志使管道使用从 PyTorch 计算的 3D 协方差而不是我们的来渲染。

</details>

2. 从新视角渲染（添加 `--novel_view`）：
```
python render.py -s data/DATASET_NAME -m output/OUTPUT_NAME -f lseg --iteration 3000 --novel_view
```
（在 `--num_views` 后添加数字以更改视图数量，例如 `--num_views 100`，默认数量为 200）

3. 使用多个插值从新视角渲染（添加 `--novel_view` 和 `--multi_interpolate`）：
```
python render.py -s data/DATASET_NAME -m output/OUTPUT_NAME -f lseg --iteration 3000 --novel_view --multi_interpolate
```

### 使用编辑渲染：
```
python render.py -s data/DATASET_NAME -m output/OUTPUT_NAME -f lseg --iteration 3000 --edit_config configs/XXX.yaml
```

### 生成视频：
运行以创建视频（添加 `--fps` 以更改 FPS，例如 `--fps 20`，默认为 10）：
```
python videos.py --data output/OUTPUT_NAME --fps 10 -f lseg  --iteration 10000
```
## 推理
### LSeg 编码器：
### 从训练好的模型分割
1. 运行以下命令使用 150 个标签进行分割（默认为 [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)）：
```shell
python -u segmentation.py --data ../../output/DATASET_NAME/ --iteration 6000
```
2. 运行以下命令使用自定义标签集进行分割（例如，添加 `--label_src car,building,tree`）：
```shell
python -u segmentation.py --data ../../output/DATASET_NAME/ --iteration 6000 --label_src car,building,tree
```

计算分割指标（用于 Replica 数据集实验，我们的预处理数据可以[在此处下载](https://drive.google.com/file/d/1sC2ZJUBRHKeWXXVUj7rIBEM-xaibvGw7/view?usp=sharing)）：
```shell
cd encoders/lseg_encoder
python -u segmentation_metric.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --student-feature-dir ../../output/OUTPUT_NAME/test/ours_30000/saved_feature/ --teacher-feature-dir ../../data/DATASET_NAME/rgb_feature_langseg/ --test-rgb-dir ../../output/OUTPUT_NAME/test/ours_30000/renders/ --workers 0 --eval-mode test
```

### SAM 编码器：
### 使用提示从训练模型的嵌入中分割（onnx）
运行以下命令（添加 `--image` 以从图像编码特征）：
1. 使用给定的输入点坐标运行（例如，添加 `--point 500 800`）：
```
python segment_prompt.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --data ../../output/OUTPUT_NAME --iteration 7000 --point 500 800
```
2. 使用给定的输入框运行（例如，添加 `--box 100 100 1500 1200`）：
```
python segment_prompt.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --data ../../output/OUTPUT_NAME --iteration 7000 --box 100 100 1500 1200
```
3. 使用给定的输入点（负）和框运行（例如，添加 `--point 500 800` 和 `--box 100 100 1500 1200`）：
```
python segment_prompt.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --data ../../output/OUTPUT_NAME --iteration 7000 --box 100 100 1500 1200 --point 500 800
```
（添加 `--onnx_path` 以更改 onnx 路径）


### 使用提示从训练模型的嵌入中分割（分割整个图像）
运行以下命令（添加 `--image` 以从图像编码特征）：
```
python segment.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --data ../../output/OUTPUT_NAME --iteration 7000
```

### 使用提示从训练模型的嵌入中分割的时间（无提示）
运行以下命令（删除 `--feature_path` 以直接从图像编码特征）：
```
python segment_time.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --image_path ../../output/OUTPUT_NAME/novel_views/ours_7000/renders/ --feature_path ../../output/OUTPUT_NAME/novel_views/ours_7000/saved_feature --output ../../output/OUTPUT_NAME
```

---

# 使用 Nerfstudio 训练

我们还提供了基于 nerfstudio 的实现，为 feature-3dgs 提供了更加模块化和高效的训练 pipeline。这种集成利用了 nerfstudio 的基础设施，同时保持了 feature-3dgs 的所有核心功能。

## 安装

### 1. 安装 nerfstudio

```bash
pip install nerfstudio
```

### 2. 安装依赖

```bash
pip install gsplat>=1.0.0
```

### 3. 克隆并包含子模块

```bash
git clone --recursive https://github.com/your-repo/feature-3dgs.git
cd feature-3dgs
```

如果您已经克隆但没有使用 `--recursive`，请运行：

```bash
git submodule update --init --recursive
```

## 快速开始

### 1. 预计算语义特征

使用 LSeg 或 SAM 从图像中提取语义特征：

```bash
python scripts/precompute_semantic_features.py \
    --data data/DATASET_NAME \
    --output data/DATASET_NAME/features \
    --model lseg \
    --resize 480 640
```

这将创建包含每张图像语义特征的 `.pt` 文件。

### 2. 注册方法

将 feature-3dgs 注册到您的 nerfstudio 安装中：

```bash
python scripts/register_feature_3dgs.py
```

或者手动添加到 nerfstudio 的 `method_configs.py` 中：

```python
import sys
from pathlib import Path
sys.path.insert(0, "path/to/feature-3dgs")

from feature_3dgs_extension.configs.feature_3dgs_configs import register_feature_3dgs_configs
register_feature_3dgs_configs(method_configs, descriptions)
```

### 3. 训练模型

```bash
# 标准训练
ns-train feature-3dgs \
    --data data/DATASET_NAME \
    --output-dir outputs/feature_3dgs_model

# 加速模式（使用 CNN 解码器加快训练速度）
ns-train feature-3dgs-speedup \
    --data data/DATASET_NAME \
    --output-dir outputs/feature_3dgs_speedup
```

### 4. 文本引导编辑

```bash
python scripts/editing_demo.py \
    --checkpoint outputs/feature_3dgs_model/nerfstudio_models/ \
    --text "椅子" \
    --operation deletion \
    --output edited_output.png
```

## 配置选项

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `--semantic-feature-dim` | 512 | 语义特征维度 |
| `--semantic-loss-weight` | 1.0 | 语义损失权重 |
| `--use-speedup` | False | 启用 CNN 解码器以加快训练 |
| `--enable-editing` | True | 启用文本引导编辑 |

## 项目结构

```
feature-3dgs/
├── feature_3dgs_extension/        # 扩展模块
│   ├── models/
│   │   └── feature_3dgs.py       # 核心模型
│   ├── data/
│   │   ├── dataparsers/
│   │   │   └── semantic_feature_dataparser.py
│   │   └── datasets/
│   │       └── semantic_feature_dataset.py
│   └── configs/
│       └── feature_3dgs_configs.py
├── third_party/
│   └── nerfstudio/               # Git 子模块
├── scripts/                       # 实用脚本
└── nerfstudio_integration/        # 文档
```

## 相关文档

- [快速入门指南](nerfstudio_integration/QUICKSTART.md)
- [实现总结](nerfstudio_integration/IMPLEMENTATION_SUMMARY.md)
- [完整文档](nerfstudio_integration/README.md)

## 致谢
我们的代码库基于 [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)、[DFFs](https://github.com/pfnet-research/distilled-feature-fields) 和 [Segment Anything](https://github.com/facebookresearch/segment-anything) 开发。非常感谢作者开源代码库。
