import torch
import torch.nn as nn
import math

class DCTMomentPooling(nn.Module):
    """
    True DCT-II based moment pooling.
    Computes mean, variance, skewness, kurtosis
    over DCT coefficients along token dimension.

    Input:  (B, N, D) - Batch, Tokens (Sequence Length), Dimensions
    Output: (B, 4D)   - Concatenated statistical moments
    """
    # Initialize the module with a small epsilon for numerical stability
    def __init__(self, eps=1e-5):
        # Call the parent class constructor
        super().__init__()
        # Store epsilon to prevent division by zero in skewness/kurtosis calculations
        self.eps = eps
        # Register a buffer to store the DCT matrix so it moves with the model to GPU/CPU
        self.register_buffer("dct_mat", None)

    # Helper function to generate the DCT-II transformation matrix
    def _build_dct(self, N, device):
        """
        Create DCT-II transform matrix of size (N, N)
        """
        # Create a column vector of indices k from 0 to N-1
        k = torch.arange(N, device=device).float().unsqueeze(1)
        # Create a row vector of indices n from 0 to N-1
        n = torch.arange(N, device=device).float().unsqueeze(0)

        # Compute the core DCT-II formula: cos(pi/N * (n + 0.5) * k)
        dct = torch.cos(math.pi / N * (n + 0.5) * k)

        # Scale the first row (DC component) by the square root of 1/N
        dct[0] *= math.sqrt(1.0 / N)
        # Scale all other rows (AC components) by the square root of 2/N
        dct[1:] *= math.sqrt(2.0 / N)

        # Return the orthogonal DCT basis matrix
        return dct

    # Define the forward pass of the pooling layer
    def forward(self, x):
        # Extract batch size (B), sequence length (N), and feature depth (D) from input
        B, N, D = x.shape

        # Check if the DCT matrix is missing or if the input sequence length has changed
        if self.dct_mat is None or self.dct_mat.shape[0] != N:
            # Rebuild the DCT matrix for the current sequence length N
            self.dct_mat = self._build_dct(N, x.device)

        # Perform matrix multiplication using Einstein notation to transform tokens to DCT domain
        # 'kn' (DCT matrix) multiplied by 'bnd' (input) results in 'bkd' (frequency coefficients)
        x_dct = torch.einsum("kn,bnd->bkd", self.dct_mat, x)

        # Calculate the first moment: the average of the DCT coefficients along the token dimension
        mean = x_dct.mean(dim=1)                
        # Subtract the mean from the DCT coefficients to center the distribution
        xc = x_dct - mean.unsqueeze(1)

        # Calculate the second moment: the average of the squared centered coefficients (variance)
        var = (xc ** 2).mean(dim=1)
        # Calculate the standard deviation by taking the square root of variance + epsilon
        std = torch.sqrt(var + self.eps)

        # Calculate the third moment (Skewness): normalized average of the cubed centered coefficients
        skew = (xc ** 3).mean(dim=1) / (std ** 3 + self.eps)
        # Calculate the fourth moment (Kurtosis): normalized average of the fourth-powered centered coefficients
        kurt = (xc ** 4).mean(dim=1) / (std ** 4 + self.eps)

        # Concatenate mean, variance, skewness, and kurtosis along the feature dimension
        # This results in a final feature vector of size (B, 4 * D)
        return torch.cat([mean, var, skew, kurt], dim=1)