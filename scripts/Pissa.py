import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.sparse.linalg import svds


class Pissa(nn.Module):
    def __init__(self, rank, W_base_tensor,bias=None):
        super().__init__()

        # Store the original bias as a non-trainable buffer
        if bias is not None:
          self.register_buffer('bias',bias)
        else:
          self.bias =None

        # Perform SVD on the W_base_tensor
        W_base_np = W_base_tensor.cpu().numpy()
        U, S, Vh = svds(W_base_np, rank, which='LM', return_singular_vectors=True)

        # Sort by descending singular values
        index = np.argsort(S)[::-1]
        S = S[index]
        U = U[:, index]
        Vh = Vh[index, :]

        # Convert to torch tensors, ensure device and dtype match original
        S=torch.from_numpy(S).to(dtype = W_base_tensor.dtype, device=device)
        U=torch.from_numpy(U).to(dtype=W_base_tensor.dtype, device=device)
        Vh=torch.from_numpy(Vh).to(dtype=W_base_tensor.dtype, device=device)

        # Residual matrix (part of the base, fixed weight)
        base_res = U @ torch.diag(S) @ Vh
        self.register_buffer('res_matrix', W_base_tensor - base_res)

        # Compute square root for initialization of LoRA A/B
        S_scaled = torch.diag(torch.sqrt(S))

        # Trainable LoRA parameters
        # LoRA_A (rank, in_features)
        self.lora_A = nn.Parameter(S_scaled @ Vh)

        # LoRA_B (out_features, rank)
        self.lora_B = nn.Parameter(U @ S_scaled)


    def forward(self, x):

        term1 = x @ self.res_matrix.T

        term2 = (x @ self.lora_A.T) @ self.lora_B.T

        if self.bias is not None:
          output = term1 + self.bias +term2
        else:
          output = term1 +term2
        return output



def Pissa_injection(model, rank, device):

  trgts = []

  for name,module in model.named_modules():
    if isinstance(module,nn.Linear):
      trgts.append((name,module))

  for name, module in trgts:
    
    W_base_tensor = module.weight

    bias = module.bias if module.bias is not None else None


    # Ensure the rank is valid for the current layer

    min_dim = min(W_base_tensor.shape)
    if min_dim <=1:
      print(f"we cannot apply SVD to layer {name} because min_dim {min_dim} is too small!")
      continue

    effective_rank = min(rank, min_dim-1)
    # Ensure effective_rank is at least 1
    effective_rank = max(1, effective_rank)
    
    pissa_layer = Pissa(effective_rank,W_base_tensor,bias)
    pissa_layer.to(device) # Move the new layer to the correct device
    modules_name = name.split(".")
    parent_module_name = modules_name[0:-1]
    parent_path = ".".join(parent_module_name)
    child_module_name = modules_name[-1]

    if parent_path:
      parent_module = model.get_submodule(parent_path)
      setattr(parent_module,child_module_name, pissa_layer)

    else:
      setattr(model, child_module_name,pissa_layer)


def setup_model(model, rank: int, use_pissa: bool, device):

    if Pissa:
      for p in model.parameters():
        p.requires_grad =False

      Pissa_injection(model, rank, device)

      trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f"Trainable parameters count: {trainable_params}")

      params = filter(lambda p: p.requires_grad, model.parameters())

    else:
        trainable_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters count: {trainable_params}")

        params = model.parameters()

    return params

