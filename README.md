# Mixture of Latent Experts (MoLE)

## Introduction
This code is a study and code reproduction of the paper "Beyond Standard MoE: Mixture of Latent Experts for Resource-Efficient Language Models". The link to the paper is: https://arxiv.org/pdf/2503.23100.
The code content is mainly divided into the following parts:
1. Establish a mathematical framework for converting a pre-trained Mixture of Experts (MoE) model into a MoLE architecture. Solve the optimization problem of the conversion through Singular Value Decomposition (SVD) to obtain the theoretically optimal solution.
2. Introduce the MoLE architecture, and decompose the expert operations into two parts: the shared low-dimensional latent space projection and the expert-specific transformation. Through matrix decomposition, decompose the expert weight matrix \(W^{i}\) into \(A^{i}B\), where \(B\) is the shared projection matrix and \(A^{i}\) is the expert-specific transformation matrix.
3. Retain the "down operator", and only convert the "up operator" and the "gating operator" from the Mixture of Experts (MoE) structure to the Mixture of Latent Experts (MoLE) structure.

## Code Explanation

### Example Code
```python
import torch


batch_size = 32
n = 100
m = 50
N = 8
k = 2
target_rank = 30


W_list = [torch.randn(n, n) for _ in range(N)]

A_up_list, B_up = transform_moe_to_mole(W_list, target_rank, m, k, N)
A_gate_list, B_gate = transform_moe_to_mole(W_list, target_rank, m, k, N)
W_down = torch.randn(m, n)


x = torch.randn(batch_size, n)


output = compute_ffn_output(x, B_up, B_gate, W_down, A_up_list, A_gate_list, k, N)
print(output.shape)
```

### Parameter Settings
In this section, we define several important parameters:
- `batch_size`: The number of samples in each batch of input data.
- `n`: The dimension of the high - dimensional space.
- `m`: The dimension of the low - dimensional space.
- `N`: The number of experts in the Mixture of Experts model.
- `k`: The group size.
- `target_rank`: The target rank for low - rank approximation.

