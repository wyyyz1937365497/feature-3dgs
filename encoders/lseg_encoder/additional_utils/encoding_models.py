###########################################################################
# Referred to: https://github.com/zhanghang1989/PyTorch-Encoding
###########################################################################
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.scatter_gather import scatter
import threading
import types
import torch
from torch.cuda._utils import _get_device_index
from torch._utils import ExceptionWrapper

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
# up_kwargs = {'mode': 'bilinear', 'align_corners': False}

__all__ = ['MultiEvalModule']


def _rebind_bound_method(module_obj, method_name):
    method = getattr(module_obj, method_name, None)
    if method is None:
        return
    method_func = getattr(method, '__func__', None)
    method_self = getattr(method, '__self__', None)
    if method_func is not None and method_self is not module_obj:
        setattr(module_obj, method_name, types.MethodType(method_func, module_obj))


def _fix_replica_monkey_patched_methods(model_replica):
    for submodule in model_replica.modules():
        _rebind_bound_method(submodule, 'forward_flex')
        _rebind_bound_method(submodule, '_resize_pos_embed')

class MultiEvalModule(DataParallel):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, device_ids=None, flip=True,
                 scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
        super(MultiEvalModule, self).__init__(module, device_ids)
        self.nclass = nclass
        self.base_size = module.base_size
        self.crop_size = module.crop_size
        self.scales = scales
        self.flip = flip
        # Tensor cache for efficient memory reuse (Phase 1.2 optimization)
        self._tensor_cache = {}
        print('MultiEvalModule: base_size {}, crop_size {}'. \
            format(self.base_size, self.crop_size))

    def _get_cached_tensor(self, shape, device, dtype=None):
        """Get cached tensor or create new one (avoids frequent allocations)

        Args:
            shape: Tensor shape tuple
            device: Target device
            dtype: Optional dtype (defaults to float32)

        Returns:
            Cached or new zero tensor
        """
        if dtype is None:
            dtype = torch.float32
        key = (shape, device, dtype)
        if key not in self._tensor_cache or self._tensor_cache[key].shape != shape:
            self._tensor_cache[key] = torch.zeros(shape, device=device, dtype=dtype)
        return self._tensor_cache[key]

    def parallel_forward(self, inputs, **kwargs):
        """Multi-GPU Mult-size Evaluation

        Args:
            inputs: list of Tensors
        """
        if not inputs:
            return []

        # CPU-only fallback path.
        if not self.device_ids:
            return [self.forward(input_tensor.unsqueeze(0), **kwargs) for input_tensor in inputs]

        max_chunk_size = max(1, len(self.device_ids))
        all_outputs = []

        # Process in chunks so batch_size can be larger than number of GPUs.
        for chunk_start in range(0, len(inputs), max_chunk_size):
            input_chunk = inputs[chunk_start:chunk_start + max_chunk_size]
            device_ids = self.device_ids[:len(input_chunk)]

            # 1) Prepare input data for this chunk.
            scattered_inputs = [
                (input_tensor.unsqueeze(0).to(device, non_blocking=True),)
                for input_tensor, device in zip(input_chunk, device_ids)
            ]

            # 2) Replicate MultiEval wrappers.
            replicas = self.replicate(self, device_ids)

            # 3) Replicate base models and bind to wrapper replicas.
            module_replicas = self.replicate(self.module, device_ids)
            for replica, module_replica, device in zip(replicas, module_replicas, device_ids):
                _fix_replica_monkey_patched_methods(module_replica)
                replica.module = module_replica
                replica.device_ids = [device]
                replica.output_device = device

            # 4) Prepare kwargs for this chunk.
            if kwargs:
                chunk_kwargs = [dict(kwargs) for _ in range(len(scattered_inputs))]
            else:
                chunk_kwargs = [{} for _ in range(len(scattered_inputs))]

            # 5) Parallel execution for this chunk.
            chunk_outputs = self.parallel_apply(replicas, scattered_inputs, chunk_kwargs)
            all_outputs.extend(chunk_outputs)

        return all_outputs
    
    def forward(self, image, return_feature=False):
        """Mult-size Evaluation"""
        # only single image is supported for evaluation
        batch, _, h, w = image.size()
        assert(batch == 1)
        stride_rate = 2.0/3.0
        crop_size = self.crop_size
        stride = int(crop_size * stride_rate)
        # Use cached tensor for scores (Phase 1.2 optimization)
        scores = self._get_cached_tensor((batch, 1, h, w), image.device).fill_(0)

        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            """
            short_size = int(math.ceil(self.base_size * scale))
            if h > w:
                width = short_size
                height = int(1.0 * h * short_size / w)
                long_size = height
            else:
                height = short_size
                width = int(1.0 * w * short_size / h)
                long_size = width
            """
            # resize image to current size
            cur_img = resize_image(image, height, width, **self.module._up_kwargs)
            if long_size <= crop_size:
                pad_img = pad_image(cur_img, self.module.mean,
                                    self.module.std, crop_size)
                outputs = module_inference(self.module, pad_img, self.flip, return_feature=return_feature) # torch.Size([1, 150, 480, 480])
                # print("###################################################### if outputs: ", outputs.shape)
                outputs = crop_image(outputs, 0, height, 0, width) # e.g. torch.Size([1, 150, 293, 390])
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = pad_image(cur_img, self.module.mean,
                                        self.module.std, crop_size)
                else:
                    pad_img = cur_img
                _,_,ph,pw = pad_img.size()
                assert(ph >= height and pw >= width)
                # Use vectorized grid inference for better performance (Phase 1.3 optimization)
                outputs = vectorized_grid_inference(
                    self.module, pad_img, crop_size, stride, self.flip, return_feature
                )
                outputs = outputs[:, :, :height, :width]
                # print("###################################################### else outputs: ", outputs.shape)

            # print("######################################################### outputs: ", outputs.shape) # e.g. torch.Size([1, 150, 293, 390]) - depend on scale 
            score = resize_image(outputs, h, w, **self.module._up_kwargs)
            # scores += score
            # print("######################################################### score: ", score.shape) # torch.Size([1, 150, 360, 480])
            scores = scores + score
        # print("######################################################### total score: ", scores.shape) # torch.Size([1, 150, 360, 480])
        return scores


def module_inference(module, image, flip=True, return_feature=False):
    with torch.amp.autocast(device_type="cuda", enabled=image.is_cuda):
        output = module.evaluate(image, return_feature=return_feature)
    if flip:
        fimg = flip_image(image)
        with torch.amp.autocast(device_type="cuda", enabled=fimg.is_cuda):
            foutput = module.evaluate(fimg, return_feature=return_feature)
        output += flip_image(foutput)
        output = output / 2
    return output


def vectorized_grid_inference(module, pad_img, crop_size, stride, flip=True, return_feature=False):
    """Vectorized grid inference for batch processing (Phase 1.3 optimization)

    Instead of processing each grid crop individually in nested loops,
    this function collects all crops and processes them in batches.

    Args:
        module: The model module
        pad_img: Padded input image [1, C, H, W]
        crop_size: Size of each grid crop
        stride: Stride between grid crops
        flip: Whether to use flip augmentation
        return_feature: Whether to return features instead of class scores

    Returns:
        Aggregated output for the entire padded image
    """
    batch, channels, ph, pw = pad_img.size()

    # Calculate grid dimensions
    h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
    w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
    total_grids = h_grids * w_grids

    # Collect all grid crops and their positions
    crops_list = []
    positions = []

    for idh in range(h_grids):
        for idw in range(w_grids):
            h0 = idh * stride
            w0 = idw * stride
            h1 = min(h0 + crop_size, ph)
            w1 = min(w0 + crop_size, pw)

            crop_img = pad_img[:, :, h0:h1, w0:w1]
            # Pad if needed (for edge crops)
            if h1 - h0 < crop_size or w1 - w0 < crop_size:
                crop_img = pad_image(crop_img, module.mean, module.std, crop_size)

            crops_list.append(crop_img)
            positions.append((h0, h1, w0, w1))

    # Process crops in conservative chunks to avoid CUDA OOM on large scenes.
    batch_size = min(8, total_grids)
    accumulated_output = None
    count_norm = None

    for batch_start in range(0, total_grids, batch_size):
        batch_end = min(batch_start + batch_size, total_grids)
        batch_crops = torch.cat([crops_list[i] for i in range(batch_start, batch_end)], dim=0)

        # Batch inference
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=batch_crops.is_cuda):
                batch_outputs = module.evaluate(batch_crops, return_feature=return_feature)

            # Handle flip augmentation
            if flip:
                batch_crops_flipped = torch.flip(batch_crops, dims=[-1])
                with torch.amp.autocast(device_type="cuda", enabled=batch_crops_flipped.is_cuda):
                    batch_outputs_flipped = module.evaluate(batch_crops_flipped, return_feature=return_feature)
                batch_outputs_flipped = torch.flip(batch_outputs_flipped, dims=[-1])
                batch_outputs = (batch_outputs + batch_outputs_flipped) / 2

        # Initialize output tensors on first batch
        if accumulated_output is None:
            # batch_outputs shape: [batch_size, num_classes_or_features, crop_h, crop_w]
            output_channels = batch_outputs.shape[1]
            accumulated_output = torch.zeros(
                batch, output_channels, ph, pw,
                device=pad_img.device, dtype=batch_outputs.dtype
            )
            count_norm = torch.zeros(
                batch, 1, ph, pw,
                device=pad_img.device, dtype=batch_outputs.dtype
            )

        # Scatter results back to their positions
        for i in range(batch_start, batch_end):
            local_idx = i - batch_start
            h0, h1, w0, w1 = positions[i]
            crop_h, crop_w = h1 - h0, w1 - w0

            # Get the output for this crop (accounting for possible padding)
            crop_output = batch_outputs[local_idx:local_idx+1, :, :crop_h, :crop_w]
            accumulated_output[:, :, h0:h1, w0:w1] += crop_output
            count_norm[:, :, h0:h1, w0:w1] += 1

    # Normalize by count
    assert (count_norm == 0).sum() == 0, "Some pixels were not covered by any grid"
    return accumulated_output / count_norm.clamp(min=1)

def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)

def pad_image(img, mean, std, crop_size):
    b,c,h,w = img.size()
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:,i,:,:] = F.pad(img[:,i,:,:], (0, padw, 0, padh), value=pad_values[i])
    assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad

def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]

def flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)
