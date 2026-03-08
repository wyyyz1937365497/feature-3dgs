import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
from modules.lseg_module import LSegModule
#from utils import Resize
from transforms_midas import Resize
import cv2
import math
import types
import functools
import torchvision.transforms as torch_transforms
import copy
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import clip
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from data import get_dataset, get_original_dataset
from additional_utils.encoding_models import MultiEvalModule as LSeg_MultiEvalModule
import torchvision.transforms as transforms
import sklearn
import sklearn.decomposition
import time
import gc

# Performance optimization modules (Phase 1-3)
try:
    from additional_utils.batch_processor import BatchFeatureExtractor, BatchConfig
    from additional_utils.async_io import AsyncIOScheduler, AsyncFeatureSaver
    from additional_utils.data_preloader import DataPreloader, create_preloading_dataloader
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("Warning: Optimization modules not available. Using original implementation.")


def test_batchify_fn(batch):
    """Collate test samples into the format expected by the evaluator."""
    images = [sample[0] for sample in batch]
    image_names = [sample[1] for sample in batch]
    return images, image_names


def get_mask_pallete(npimg, palette):
    """Create a paletted PIL image from a predicted label map."""
    out_img = Image.fromarray(npimg.squeeze().astype("uint8"))
    out_img.putpalette(palette)
    return out_img


def is_cuda_alloc_error(exc):
    """Detect CUDA allocation failures that can be mitigated by fallback settings."""
    msg = str(exc).lower()
    tokens = (
        "out of memory",
        "cublas_status_alloc_failed",
        "cuda error: cublas",
    )
    return any(token in msg for token in tokens)


def clear_cuda_memory():
    """Best-effort CUDA memory cleanup before retrying a failed batch."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            # ipc_collect may fail on some drivers/runtime combinations.
            pass


def resize_images(image_list, target_h, target_w):
    """Resize a list of CHW tensors to a shared target size."""
    if image_list[0].shape[-2:] == (target_h, target_w):
        return image_list
    return [
        F.interpolate(
            img[None],
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=True,
        )[0]
        for img in image_list
    ]

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
        # model and dataset
        parser.add_argument(
            "--model", type=str, default="encnet", help="model name (default: encnet)"
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone name (default: resnet50)",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            # default="ade20k",
            default="ignore",
            help="dataset name (default: pascal12)",
        )
        parser.add_argument(
            "--workers", type=int, default=16, metavar="N", help="dataloader threads"
        )
        parser.add_argument(
            "--base-size", type=int, default=520, help="base image size"
        )
        parser.add_argument(
            "--crop-size", type=int, default=480, help="crop image size"
        )
        parser.add_argument(
            "--train-split",
            type=str,
            default="train",
            help="dataset train split (default: train)",
        )
        # training hyper params
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument(
            "--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)"
        )
        parser.add_argument(
            "--train-batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            training (default: auto)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            testing (default: same as batch size)",
        )
        # cuda, seed and logging
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        parser.add_argument(
            "--weights", type=str, default=None, help="checkpoint to test"
        )
        parser.add_argument(
            "--eval", action="store_true", default=False, help="evaluating mIoU"
        )
        parser.add_argument(
            "--export",
            type=str,
            default=None,
            help="put the path to resuming file if needed",
        )

        parser.add_argument(
            "--acc-bn",
            action="store_true",
            default=False,
            help="Re-accumulate BN statistics",
        )
        parser.add_argument(
            "--test-val",
            action="store_true",
            default=False,
            help="generate masks on val set",
        )
        parser.add_argument(
            "--no-val",
            action="store_true",
            default=False,
            help="skip validation during training",
        )
        parser.add_argument(
            "--module",
            default='lseg',
            help="select model definition",
        )
        # test option
        parser.add_argument(
            "--data-path", type=str, default=None, help="path to test image folder"
        )
        parser.add_argument(
            "--no-scaleinv",
            dest="scale_inv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )
        parser.add_argument(
            "--widehead", default=False, action="store_true", help="wider output head"
        )
        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )
        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )
        parser.add_argument(
            "--label_src",
            type=str,
            default="default",
            help="how to get the labels",
        )
        parser.add_argument(
            "--jobname",
            type=str,
            default="default",
            help="select which dataset",
        )
        parser.add_argument(
            "--no-strict",
            dest="strict",
            default=True,
            action="store_false",
            help="no-strict copy the model",
        )
        parser.add_argument(
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )
        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )
        parser.add_argument(
            "--activation",
            choices=['lrelu', 'tanh'],
            default="lrelu",
            help="use which activation to activate the block",
        )

        parser.add_argument(
            "--outdir",
            default="outdir_ours",
            help="output direcory of features",
        )
        parser.add_argument(
            "--test-rgb-dir",
            help="test rgb dir",
            required=True,
        )

        parser.add_argument(
            "--resize-max", type=float, default=1.25, help=""
        )

        # Performance optimization arguments (Phase 1-3)
        parser.add_argument(
            "--optimize-level",
            type=int,
            default=0,
            choices=[0, 1, 2, 3],
            help="Optimization level: 0=original, 1=quick wins (CPU-GPU, caching), 2=with batch+async I/O, 3=full optimization"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=4,
            help="Batch size for level >= 2 optimization"
        )

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args

adepallete = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,255,214,0,25,194,194,102,255,0,92,0,255]

    
import matplotlib.patches as mpatches
def get_legend_patch(npimg, new_palette, labels):
    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    out_img.putpalette(new_palette)
    u_index = np.unique(npimg)
    patches = []
    for i, index in enumerate(u_index):
        label = labels[index]
        cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0, new_palette[index * 3 + 2] / 255.0]
        red_patch = mpatches.Patch(color=cur_color, label=label)
        patches.append(red_patch)
    return out_img, patches

def test(args):

    module = LSegModule.load_from_checkpoint(
        checkpoint_path=args.weights,
        data_path=args.data_path,
        dataset=args.dataset,
        backbone=args.backbone,
        aux=args.aux,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=args.ignore_index,
        dropout=0.0,
        scale_inv=args.scale_inv,
        augment=False,
        no_batchnorm=False,
        widehead=args.widehead,
        widehead_hr=args.widehead_hr,
        map_locatin="cpu",
        arch_option=args.arch_option,
        strict=args.strict,
        block_depth=args.block_depth,
        activation=args.activation,
    )
    labels = module.get_labels('ade20k')
    input_transform = module.val_transform
    # num_classes = module.num_classes
    num_classes = len(labels)
    #labels.append("petal")
    #labels.append("leaf")
    #num_classes += 2

    # dataset
    print("test rgb dir", args.test_rgb_dir)
    # testset = get_dataset(
    testset = get_original_dataset(
        # "/mnt/nfs-mnj-archive-02/user/sosk/neural_semantic_field/semantic_nerf/Replica_Dataset/room_0/Sequence_1/rgb",
        #"/mnt/nfs-mnj-archive-02/user/sosk/neural_semantic_field/semantic_nerf/Replica_Dataset/office_0/Sequence_1/rgb",
        #"/mnt/nfs-mnj-hot-01/tmp/sosk/neural_semantic_field/semantic_nerf/Replica_Dataset/office_0/Sequence_1/rgb",
        args.test_rgb_dir,
        #args.dataset,
        #root=args.data_path,
        #split="val",
        #mode="testval" if args.eval else "test",  # test returns (image, path), the others returns (image, GTmask)
        transform=input_transform,
    )

    # dataloader
    loader_kwargs = (
        {"num_workers": args.workers, "pin_memory": True} if args.cuda else {}
    )
    test_data = data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=test_batchify_fn,
        **loader_kwargs
    )

    model = module

    model = model.eval()
    # Remove unnecessary CPU transfer - model will be moved to GPU by evaluator

    print(model)

    if args.export:
        torch.save(model.state_dict(), args.export + ".pth")
        return

    """
    scales = (
        [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        if args.dataset == "citys"
        # else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        # else np.linspace(1.0, 3.0, 6)
        #else np.linspace(0.5, 4.0, 6)
        #else np.linspace(0.75, 1.25, 6)
        #else np.linspace(2.0, 4.0, 6)
        #else np.linspace(1.0, 2.0, 6)
        # else np.linspace(0.75, 3.0, 6)
        # else np.linspace(0.75, 2.25, 7)
        else np.linspace(0.75, args.resize_max, 7)
        ## else np.linspace(0.75, 1.75, 7)
        # else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    )
    """
    # scales = np.linspace(0.75, 2.25, 7)
    # scales = np.linspace(0.75, args.resize_max, 7)
    # scales = [0.75, 1.0, 1.25]
    # scales = [0.75, 1.0, 1.25, 1.5, 1.75]
    scales = [0.75, 1.0, 1.25, 1.75]
    print("base scales", scales)
    print("test rgb dir", args.test_rgb_dir)
    print("outdir", args.outdir)

    tbar = tqdm(test_data)

    f = open("log_test_{}_{}.txt".format(args.jobname, args.dataset), "a+")
    per_class_iou = np.zeros(testset.num_class)
    print(testset.num_class)
    cnt = 0

    if "Replica_Dataset" in args.test_rgb_dir:
        print(args.data_path, "is Replica_Dataset. So, skip some frames.")
        total_num = 900
        step = 5
        train_ids = list(range(0, total_num, step))
        test_ids = [x+step//2 for x in train_ids]
        assert len(testset) == total_num, (len(testset), total_num)
        assert args.test_batch_size == 1, args.test_batch_size

    # output folder
    # outdir = "outdir_ours"
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if "Replica_Dataset" in args.test_rgb_dir:
        w, h = 320, 240
    elif "scannnet" in args.test_rgb_dir:
        w, h = 320, 240
    else:
        w, h = 480, 360
    print("w, h =", w, h)

    base_h, base_w = h, w
    lowres_w = min(base_w, 384)
    lowres_h = max(240, int(round(base_h * lowres_w / float(base_w))))

    # Use stable single_scale profile as default (user preference: fast and reliable)
    runtime_profiles = [
        {
            "name": "single_scale",
            "scales": [1.0],
            "flip": False,
            "size": (base_h, base_w),
        },
    ]
    if (lowres_h, lowres_w) != (base_h, base_w):
        runtime_profiles.append(
            {
                "name": "single_scale_lowres",
                "scales": [1.0],
                "flip": False,
                "size": (lowres_h, lowres_w),
            }
        )

    def build_evaluator(profile):
        evaluator_obj = LSeg_MultiEvalModule(
            model,
            num_classes,
            scales=profile["scales"],
            flip=profile["flip"],
        )
        if args.cuda:
            evaluator_obj = evaluator_obj.cuda()
        evaluator_obj.eval()
        return evaluator_obj

    active_profile_idx = 0
    active_profile = runtime_profiles[active_profile_idx]
    h, w = active_profile["size"]
    evaluator = build_evaluator(active_profile)
    print(
        f"Active evaluator profile '{active_profile['name']}' "
        f"(size={h}x{w}, scales={active_profile['scales']}, flip={active_profile['flip']})"
    )

    # Initialize async I/O scheduler for level >= 2
    io_scheduler = None
    if args.optimize_level >= 2 and OPTIMIZATION_AVAILABLE:
        print("Using async I/O for non-blocking file operations")
        io_scheduler = AsyncIOScheduler(max_workers=4, queue_size=32)

    pca = None
    print("test rgb dir", args.test_rgb_dir)
    print("outdir", args.outdir)

    def run_batch_with_retry(raw_images):
        nonlocal evaluator, active_profile, active_profile_idx, h, w

        while True:
            prepared_images = raw_images
            if prepared_images[0].shape[-1] > w or prepared_images[0].shape[-2] > h:
                print("resize", prepared_images[0].shape, "to", (h, w))
                prepared_images = resize_images(prepared_images, h, w)
                print(prepared_images[0].shape)

            try:
                outputs = evaluator.parallel_forward(prepared_images)
                output_features = evaluator.parallel_forward(prepared_images, return_feature=True)
                return outputs, output_features, prepared_images
            except RuntimeError as exc:
                if not is_cuda_alloc_error(exc):
                    raise

                if active_profile_idx + 1 >= len(runtime_profiles):
                    raise RuntimeError(
                        "CUDA memory allocation failed even after all fallback profiles. "
                        f"Last profile='{active_profile['name']}', "
                        f"size={h}x{w}, scales={active_profile['scales']}, flip={active_profile['flip']}"
                    ) from exc

                print(
                    "[Warn] CUDA allocation failed in profile "
                    f"'{active_profile['name']}'. Switching to fallback profile and retrying image."
                )
                active_profile_idx += 1
                active_profile = runtime_profiles[active_profile_idx]
                h, w = active_profile["size"]
                clear_cuda_memory()
                evaluator = build_evaluator(active_profile)
                print(
                    f"[Warn] Active evaluator profile '{active_profile['name']}' "
                    f"(size={h}x{w}, scales={active_profile['scales']}, flip={active_profile['flip']})"
                )
    for i, (image, dst) in enumerate(tbar):
        """
        if "Replica_Dataset" in args.test_rgb_dir and not (i in train_ids or i in test_ids):
            impath = dst[0]
            print("save dummy array for", impath)
            fmap = np.zeros(1)  # dummy
            np.savez_compressed(os.path.join(outdir, os.path.splitext(impath)[0] + "_fmap__ori_w{}xh{}.npz".format(w, h)), fmap)
            continue
        """

        with torch.no_grad():
            raw_images = image
            print(raw_images[0].shape, "image.shape -")
            outputs, output_features, image = run_batch_with_retry(raw_images)
            print(image[0].shape, "image.shape")
            print("start pred")
            start = time.time()
            # print(output_features.shape, output_features.min(), output_features.max())
            # print(type(outputs), type(output_features))
            print("done pred", time.time() - start)
            # list
            print("start make_pred")
            start = time.time()
            predicts = [
                testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
                for output in outputs
            ]
            del outputs
            print("done makepred", time.time() - start)
            # output_features = [o.cpu().numpy().astype(np.float16) for o in output_features]

        for predict, impath, img, fmap in zip(predicts, dst, image, output_features):
            # prediction mask
            # mask = utils.get_mask_pallete(predict - 1, args.dataset)
            mask = get_mask_pallete(predict - 1, adepallete)
            outname = os.path.splitext(impath)[0] + ".png"

            # Use async I/O for level >= 2
            if io_scheduler is not None:
                io_scheduler.submit_save(
                    np.array(mask),
                    os.path.join(outdir, outname),
                    save_fn=lambda data, path: Image.fromarray(data).save(path)
                )
            else:
                mask.save(os.path.join(outdir, outname))

            # vis from accumulation of prediction
            mask = torch.tensor(np.array(mask.convert("RGB"), "f")) / 255.0
            vis_img = (img + 1) / 2.
            vis_img = vis_img.permute(1, 2, 0)  # ->hwc
            vis1 = vis_img
            vis2 = vis_img * 0.4 + mask * 0.6
            vis3 = mask
            vis = torch.cat([vis1, vis2, vis3], dim=1)

            if io_scheduler is not None:
                io_scheduler.submit_save(
                    (vis.cpu().numpy() * 255).astype(np.uint8),
                    os.path.join(outdir, outname + "_vis.png"),
                    save_fn=lambda data, path: Image.fromarray(data).save(path)
                )
            else:
                Image.fromarray((vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(outdir, outname + "_vis.png"))

            # new_palette = get_new_pallete(len(labels))
            # seg, patches = get_new_mask_pallete(predict, new_palette, labels=labels)
            # print(predict.min(), predict.max())
            seg, patches = get_legend_patch(predict - 1, adepallete, labels)
            seg = seg.convert("RGBA")
            plt.figure()
            plt.axis('off')
            plt.imshow(seg)
            #plt.legend(handles=patches)
            plt.legend(handles=patches, prop={'size': 8}, ncol=4)

            # 异步保存图例图像
            legend_path = os.path.join(outdir, outname + "_legend.png")
            plt.savefig(legend_path, format="png", dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close()

            ###############################
            # print(fmap.shape)  # torch.Size([1, 512, 512, 683])
            #print(fmap.shape, h, w)
            start = time.time()
            ###
            # save unnormalized image feature as compressed npz (节省存储空间)
            unnormalized_fmap = fmap[0]  # [512, h, w]
            feature_np = unnormalized_fmap.cpu().numpy().astype(np.float16)
            output_base = os.path.join(outdir, os.path.splitext(impath)[0])

            if io_scheduler is not None:
                # Async save for feature map (npz compressed) - 不使用自定义 save_fn，让 async_io 处理
                io_scheduler.submit_save(
                    feature_np,
                    output_base + "_fmap_CxHxW.npz"
                )
            else:
                # Sync save as npz compressed
                np.savez_compressed(output_base + "_fmap_CxHxW.npz", features=feature_np)

            fmap = F.interpolate(fmap, size=(h, w), mode='bilinear', align_corners=False)  # [1, 512, h, w]
            fmap = F.normalize(fmap, dim=1)  # normalize
            #print(time.time() - start)
            #print("done interpolate")

            if pca is None:
                print("calculate PCA based on 1st image", impath)
                pca = sklearn.decomposition.PCA(3, random_state=42)
                f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
                transformed = pca.fit_transform(f_samples)
                print(pca)
                print("pca.explained_variance_ratio_", pca.explained_variance_ratio_.tolist())
                print("pca.singular_values_", pca.singular_values_.tolist())
                feature_pca_mean = torch.tensor(f_samples.mean(0)).float()
                feature_pca_components = torch.tensor(pca.components_).float()
                q1, q99 = np.percentile(transformed, [1, 99])
                feature_pca_postprocess_sub = q1
                feature_pca_postprocess_div = (q99 - q1)
                print(q1, q99)
                del f_samples
                pca_dict = {"pca": pca, "feature_pca_mean": feature_pca_mean, "feature_pca_components": feature_pca_components,
                            "feature_pca_postprocess_sub": feature_pca_postprocess_sub, "feature_pca_postprocess_div": feature_pca_postprocess_div}
                pca_path = os.path.join(outdir, "pca_dict.pt")

                # 异步保存PCA参数
                if io_scheduler is not None:
                    io_scheduler.submit_save(
                        pca_dict,
                        pca_path,
                        save_fn=lambda data, path: torch.save(data, path)
                    )
                else:
                    torch.save(pca_dict, pca_path)

                # Move PCA parameters to GPU once after initialization
                feature_pca_mean = feature_pca_mean.to(fmap.device)
                feature_pca_components = feature_pca_components.to(fmap.device)

            #print("start imgsave")
            start = time.time()
            # Keep PCA buffers on the same device as the current fmap replica.
            if feature_pca_mean.device != fmap.device:
                feature_pca_mean = feature_pca_mean.to(fmap.device, non_blocking=True)
                feature_pca_components = feature_pca_components.to(fmap.device, non_blocking=True)
            vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
            vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
            vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3))
            # Single CPU transfer at the end for numpy conversion
            if io_scheduler is not None:
                io_scheduler.submit_save(
                    (vis_feature.cpu().numpy() * 255).astype(np.uint8),
                    os.path.join(outdir, outname + "_feature_vis.png"),
                    save_fn=lambda data, path: Image.fromarray(data).save(path)
                )
            else:
                Image.fromarray((vis_feature.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(outdir, outname + "_feature_vis.png"))
            #print(time.time() - start)
            #print("done imgsave")

    # Wait for all async I/O to complete
    if io_scheduler is not None:
        print("Waiting for async I/O to complete...")
        io_scheduler.wait_completion()
        stats = io_scheduler.get_stats()
        print(f"I/O Stats: {stats.successful_saves} successful, {stats.failed_saves} failed")
        if stats.failed_saves > 0:
            print("Errors encountered:")
            for error in io_scheduler.get_errors()[:5]:  # Show first 5 errors
                print(f"  {error}")
        io_scheduler.shutdown()
class ReturnFirstClosure(object):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        outputs = self._data[idx]
        return outputs[0]


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)

    # Adjust batch size based on optimization level
    if args.optimize_level >= 2 and args.batch_size > 0:
        args.test_batch_size = args.batch_size
    else:
        args.test_batch_size = 1

    print(f"Running with optimization level {args.optimize_level}")
    if args.optimize_level >= 2:
        print(f"Using batch size: {args.test_batch_size}")

    test(args)