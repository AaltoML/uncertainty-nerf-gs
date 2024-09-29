from typing import Optional, Tuple
from torch import nn
import torch


def create_mlp(
    in_dim: int,
    num_layers: int,
    layer_width: int,
    out_dim: int,
    skip_connections: Optional[Tuple[int]] = None,
    activation: Optional[nn.Module] = nn.ReLU,
    out_activation: nn.Module = None,
    dropout_layers: Optional[Tuple[float]] = None,
    dropout_rate: Optional[float] = None,
    dtype: torch.dtype = torch.float32
):
    layers = []
    skip_connections = set(skip_connections) if skip_connections else set()
    dropout_layers = set(dropout_layers) if dropout_layers else set()
    
    if num_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim, dtype=dtype))
    else:
        for i in range(num_layers - 1):
            if i in dropout_layers:
                layers.append(nn.Dropout(p=dropout_rate))
            if i == 0:
                assert i not in skip_connections, "No skip connection for layer 0"
                layers.append(nn.Linear(in_dim, layer_width, dtype=dtype))
            elif i in skip_connections:
                layers.append(nn.Linear(in_dim + layer_width, layer_width, dtype=dtype))
            else:
                layers.append(nn.Linear(layer_width, layer_width, dtype=dtype))
            if activation:
                layers.append(activation())
        if (num_layers - 1) in dropout_layers or( -1 in dropout_layers):
            layers.append(nn.Dropout(p=dropout_rate))

        layers.append(nn.Linear(layer_width, out_dim, dtype=dtype))
        if out_activation:
            layers.append(out_activation())
    return nn.Sequential(*layers)