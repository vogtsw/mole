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

# Parameter settings
# Batch size of the input data.
batch_size = 32
# Dimension of the high-dimensional space.
n = 100
# Dimension of the low-dimensional space.
m = 50
# Number of experts.
N = 8
# Group size.
k = 2
# Target rank for low-rank approximation.
target_rank = 30

# Initialize MoE expert weights, each matrix with shape (n, n).
W_list = [torch.randn(n, n) for _ in range(N)]

# Convert to MoLE parameters.
# Compute the list of experts' specific up-projection transformation matrices and the list of shared up-projection latent mapping matrices.
A_up_list, B_up = transform_moe_to_mole(W_list, target_rank, m, k, N)
# Similarly, compute the matrices related to gating.
A_gate_list, B_gate = transform_moe_to_mole(W_list, target_rank, m, k, N)

# Initialize the down-projection matrix with shape (m, n).
W_down = torch.randn(m, n)

# Generate random input data with shape (batch_size, n).
x = torch.randn(batch_size, n)

# Compute the output of the FFN layer.
output = compute_ffn_output(x, B_up, B_gate, W_down, A_up_list, A_gate_list, k, N)
# Print the shape of the output, which should be torch.Size([32, 100]).
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

