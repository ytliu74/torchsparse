import argparse
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.cuda import amp

import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.nn import functional as F
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize


def from_dense(x: torch.Tensor):
    """create sparse tensor fron channel last dense tensor by to_sparse
    x must be BTHWC tensor, channel last
    """
    sparse_data = x.to_sparse(x.ndim - 1)
    spatial_shape = sparse_data.shape[:-1]
    sparse_indices = sparse_data.indices().transpose(1, 0).contiguous().int()
    sparse_feature = sparse_data.values()

    return SparseTensor(
        feats=sparse_feature.cuda(),
        coords=sparse_indices.cuda(),
        spatial_range=spatial_shape,
    )


if __name__ == "__main__":
    conv_config = F.conv_config.get_default_conv_config()
    # conv_config.dataflow = F.Dataflow.GatherScatter
    F.conv_config.set_global_conv_config(conv_config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp_enabled", action="store_true")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    BS = 2
    T = 1
    W = 10
    H = 10
    C = 5

    map = torch.rand(BS, T, W, H, C).to(args.device)
    sparsity = 0.5
    mask = map.mean(dim=-1, keepdim=True) < sparsity
    map_masked = map * mask.float()

    print(map.shape)

    sparse_map = from_dense(map_masked)
    print(sparse_map.C.shape, sparse_map.F.shape)
    print(sparse_map.spatial_range)

    dense_map = sparse_map.dense()

    # Check whether the dense map is the same as the original map
    print((dense_map - map_masked).abs().max())


    model = nn.Sequential(
        spnn.Conv3d(C, 32, kernel_size=(1, 3, 3), stride=2, padding=(0, 1, 1)),
        spnn.BatchNorm(32),
        spnn.ReLU(True),
    ).to(args.device)

    output = model(sparse_map)
    print(output.C.shape, output.F.shape)
    print(output.spatial_range)
    print(output.dense().shape)
