import torch
import torch.nn as nn

def compute_expert_i(x, B_up, B_gate, W_down, A_up_i, A_gate_i, k, i):
    """
    Compute the output of the i-th expert.
    :param x: Input tensor with shape (batch_size, n).
    :param B_up: List of shared up-projection latent mapping matrices with shape (G, m, n), where G = N // k.
    :param B_gate: List of shared gating latent mapping matrices with shape (G, m, n), where G = N // k.
    :param W_down: Down-projection matrix with shape (m, n).
    :param A_up_i: The i-th expert's specific up-projection transformation matrix with shape (m, m).
    :param A_gate_i: The i-th expert's specific gating transformation matrix with shape (m, m).
    :param k: Group size.
    :param i: Expert index.
    :return: Output tensor of the i-th expert with shape (batch_size, n).
    """
    # Calculate the group index to select the corresponding matrix from the shared matrix list.
    idx = i // k
    # Get the shared up-projection matrix corresponding to the current group, with shape (m, n).
    B_up_idx = B_up[idx]
    # Transpose the input tensor for matrix multiplication, resulting in shape (n, batch_size).
    x_transposed = x.T
    # Compute the matrix multiplication of B_up_idx and x_transposed, resulting in shape (m, batch_size).
    B_up_x = torch.matmul(B_up_idx, x_transposed)
    # Compute the matrix multiplication of A_up_i and B_up_x, and then transpose the result back to (batch_size, m).
    term1 = torch.matmul(A_up_i, B_up_x).T

    # Get the shared gating matrix corresponding to the current group, with shape (m, n).
    B_gate_idx = B_gate[idx]
    # Compute the matrix multiplication of B_gate_idx and x_transposed, resulting in shape (m, batch_size).
    B_gate_x = torch.matmul(B_gate_idx, x_transposed)
    # Compute the matrix multiplication of A_gate_i and B_gate_x, apply the ReLU activation function, and then transpose the result back to (batch_size, m).
    act_term = torch.relu(torch.matmul(A_gate_i, B_gate_x).T)

    # Compute the element-wise multiplication of term1 and act_term, resulting in shape (batch_size, m).
    inner_product = term1 * act_term
    # Compute the matrix multiplication of inner_product and W_down to get the final output with shape (batch_size, n).
    result = torch.matmul(inner_product, W_down)
    return result

def compute_ffn_output(x, B_up, B_gate, W_down, A_up_list, A_gate_list, k, N):
    """
    Compute the output of the FFN layer using the MoLE architecture.
    :param x: Input tensor with shape (batch_size, n).
    :param B_up: List of shared up-projection latent mapping matrices with shape (G, m, n), where G = N // k.
    :param B_gate: List of shared gating latent mapping matrices with shape (G, m, n), where G = N // k.
    :param W_down: Down-projection matrix with shape (m, n).
    :param A_up_list: List containing all experts' specific up-projection transformation matrices, each with shape (m, m).
    :param A_gate_list: List containing all experts' specific gating transformation matrices, each with shape (m, m).
    :param k: Group size.
    :param N: Number of experts.
    :return: Output tensor of the FFN layer with shape (batch_size, n).
    """
    # Initialize the accumulator with the same shape as the input x.
    sum_term = torch.zeros_like(x)
    # Iterate over all experts.
    for i in range(N):
        # Compute the output of the i-th expert.
        expert_output = compute_expert_i(x, B_up, B_gate, W_down, A_up_list[i], A_gate_list[i], k, i)
        # Add the output of the i-th expert to the sum_term.
        sum_term += expert_output
    # The final output is the sum of the input x and the accumulator.
    return x + sum_term

def transform_moe_to_mole(W_list, target_rank, m, k, N):
    """
    Transform the MoE (Mixture of Experts) architecture to the MoLE (Mixture of Latent Experts) architecture.
    :param W_list: List containing all MoE expert weight matrices, each with shape (n, n).
    :param target_rank: Target rank for low-rank approximation.
    :param m: Latent dimension.
    :param k: Group size.
    :param N: Number of experts.
    :return: A_up_list: List containing all experts' specific up-projection transformation matrices, each with shape (m, m).
             B_up: List of shared up-projection latent mapping matrices with shape (G, m, n), where G = N // k.
    """
    # Calculate the number of groups.
    G = N // k
    # Get the dimension of the expert weight matrix.
    n = W_list[0].shape[0]
    # Initialize the list of shared up-projection latent mapping matrices.
    B_up = []
    # Initialize the list of experts' specific up-projection transformation matrices.
    A_up_list = []

    # Step 1: Perform low-rank approximation for each expert.
    W_tilde_list = []
    for W in W_list:
        # Perform singular value decomposition on each expert weight matrix.
        U, S, V = torch.svd(W)
        # Truncate the singular values and vectors for low-rank approximation.
        W_tilde = U[:, :target_rank] @ torch.diag(S[:target_rank]) @ V[:, :target_rank].T
        # Add the low-rank approximated matrix to the list.
        W_tilde_list.append(W_tilde)

    # Step 2: Group decomposition.
    for j in range(G):
        # Select the matrices of the current group from the low-rank approximated matrix list and stack them into a 3D tensor with shape (k, n, n).
        group_W = torch.stack(W_tilde_list[j*k:(j+1)*k], dim=0)
        # Flatten the weight matrices of the experts in the group into a 2D tensor with shape (k*n, n).
        M = group_W.view(-1, n)
        # Perform singular value decomposition on the flattened matrix.
        U, S, V = torch.svd(M)

        # Compute the shared B matrix with shape (m, n).
        B_j = (torch.diag(S[:m]) @ V[:, :m].T).view(m, n)
        # Add the shared B matrix to the list.
        B_up.append(B_j)

        # Compute the experts' specific A matrices.
        A_group = U[:, :m] @ torch.diag(torch.sqrt(S[:m]))
        # Reshape A_group into a 3D tensor with shape (k, n, m).
        A_group = A_group.view(k, n, m)

        for i in range(k):
            # Example decomposition to generate an A matrix with shape (m, m).
            A_i = A_group[i].T @ A_group[i]
            # Add the experts' specific A matrix to the list.
            A_up_list.append(A_i)

    # Stack the list of shared B matrices into a 3D tensor with shape (G, m, n).
    B_up = torch.stack(B_up, dim=0)
    return A_up_list, B_up

