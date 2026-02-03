import torch
import math
import torch.nn as nn
import torch.nn.functional as F




'''
class TransformerAttention(nn.Module):
    def __init__(self, input_dim, output_dim, last_dim=1):
        super(TransformerAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim*output_dim, last_dim)
        # self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.unsqueeze(dim=1)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1)))
        attention = self.softmax(attention)
        output = torch.matmul(attention, v)
        output = output.view([output.shape[0], -1])
        output = self.fc(output)
        return output
'''   
# For training

class TransformerAttention(nn.Module):
    """
    Self-Attention module over multiple branches (original, patch-shuffle, noise).

    Args:
        input_dim (int): Dimension of each feature vector (e.g., 1024 for ViT-L/14 penultimate)
        num_branches (int): Number of branches stacked together
        last_dim (int): Output dimension (e.g., number of classes)
    """
    def __init__(self, input_dim, num_branches, last_dim=1):
        super(TransformerAttention, self).__init__()
        self.input_dim = input_dim
        self.num_branches = num_branches
        self.last_dim = last_dim

        # Linear layers for query, key, value
        self.query = nn.Linear(input_dim, input_dim)
        self.key   = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # Fully connected layer maps flattened attention output to last_dim
        self.fc = nn.Linear(input_dim * num_branches, last_dim)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x: Tensor of shape (B, num_branches, input_dim)
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compute attention weights
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1), device=x.device, dtype=x.dtype))
        attention = self.softmax(attention)

        # Weighted sum of values
        output = torch.matmul(attention, v)  # (B, num_branches, input_dim)

        # Flatten branches
        output = output.view(output.size(0), -1)  # (B, num_branches * input_dim)

        # Final linear layer
        output = self.fc(output)  # (B, last_dim)
        return output



'''
class TransformerAttention(nn.Module):
    """
    CLIP-style Multi-Head Self-Attention adapted for multiple branches (original, patch-shuffle, noise).
    Compatible with CLIP checkpoints.
    """
    def __init__(self, input_dim, num_branches, last_dim=1, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.num_branches = num_branches
        self.last_dim = last_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        # CLIP-style projections
        self.in_proj_weight = nn.Parameter(torch.empty(3 * input_dim, input_dim))
        self.in_proj_bias   = nn.Parameter(torch.empty(3 * input_dim))

        self.out_proj = nn.Linear(input_dim, input_dim)

        # Final classification layer
        self.fc = nn.Linear(input_dim * num_branches, last_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, num_branches, input_dim)
        Returns:
            Tensor of shape (B, last_dim)
        """
        B, N, C = x.shape  # (batch, branches, input_dim)

        # Apply in-projection (q, k, v)
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)  # each: (B, N, input_dim)

        # Split into heads
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, heads, N, head_dim)

        # Recombine heads
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, input_dim)

        # Output projection
        out = self.out_proj(out)  # (B, N, input_dim)

        # Flatten branches
        out = out.reshape(B, -1)  # (B, num_branches * input_dim)

        # Final classifier
        return self.fc(out)
'''



class TransformerMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, head_num, predict_dim=1):
        super(TransformerMultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.query = nn.Linear(input_dim, input_dim * head_num)
        self.key = nn.Linear(input_dim, input_dim * head_num)
        self.value = nn.Linear(input_dim, input_dim * head_num)
        self.fc = nn.Linear(input_dim * head_num * output_dim, predict_dim)
        self.softmax = nn.Softmax(dim=1)

    def split_heads(self, tensor):
        # Split the last dimension into (head_num, new_last_dim)
        tensor = tensor.view(tensor.size(0), tensor.size(1), self.head_num, tensor.size(-1) // self.head_num)
        # Transpose the result to (batch, head_num, new_last_dim, -1)
        return tensor.transpose(1, 2)
    

    def scaled_dot_product_attention(self, q, k, v):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention = self.softmax(scores)
        return torch.matmul(attention, v)

    def combine_heads(self, tensor):
        # Transpose and reshape the tensor to (batch, -1, head_num * new_last_dim)
        return tensor.transpose(1, 2).contiguous().view(tensor.size(0), -1)
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q, k, v = [self.split_heads(tensor) for tensor in (q, k, v)]
        attention = self.scaled_dot_product_attention(q, k, v)
        output = self.combine_heads(attention)
        output = self.fc(output)
        return output



class TransformerAttentionwithClassifierToken(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim):
        super(TransformerAttentionwithClassifierToken, self).__init__()
        self.input_dim = input_dim
        self.classifier_token = nn.Linear(input_dim, 1)
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        classifier_token = torch.ones((x.shape[0], 1, self.input_dim)).to(x.device)*self.classifier_token.weight.data.squeeze(dim=-1)
        x = torch.cat([classifier_token.to(x.device), x], dim=-2)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1)))
        attention = self.softmax(attention)
        output = torch.matmul(attention, v)
        # using only the classifier token
        output = self.fc(output[:, 0, :])
        return output


class TransformerCrossAttention(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim):
        super(TransformerAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim*middle_dim, output_dim)
        # self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.unsqueeze(dim=1)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1)))
        attention = self.softmax(attention)
        output = torch.matmul(attention, v)
        output = output.view([output.shape[0], -1])
        output = self.fc(output)
        return output





class TransformerAttentionwithPisition(nn.Module):
    def __init__(self, input_dim, output_dim, last_dim=1, token_num=2):
        super(TransformerAttentionwithPisition, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim * token_num, last_dim)
        self.softmax = nn.Softmax(dim=1)
        self.position_embeddings = nn.Parameter(torch.zeros([token_num, input_dim]))

    def forward(self, x):
        seq_len = x.size(1)
        x_with_position = x + self.position_embeddings

        q = self.query(x_with_position)
        k = self.key(x_with_position)
        v = self.value(x_with_position)

        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))

        attention = self.softmax(attention)
        output = torch.matmul(attention, v)

        output = output.view([output.shape[0], -1])
        output = self.fc(output)
        return output



class TransformerCrossAttentionwithPisition(nn.Module):
    def __init__(self, input_dim, output_dim, last_dim=1, token_num=2):
        super(TransformerCrossAttentionwithPisition, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim * output_dim // 2, last_dim)
        self.softmax = nn.Softmax(dim=1)
        self.position_embeddings = nn.Parameter(torch.zeros([token_num, input_dim]))

    def forward(self, x):
        # first half to be q and second half to be k, v
        seq_len = x.size(1)
        x_with_position = x + self.position_embeddings

        q = self.query(x_with_position[:, :x_with_position.shape[1]//2, :])
        k = self.key(x_with_position[:, x_with_position.shape[1]//2:, :])
        v = self.value(x_with_position[:, x_with_position.shape[1]//2:, :])

        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))

        attention = self.softmax(attention)
        output = torch.matmul(attention, v)

        output = output.view([output.shape[0], -1])
        output = self.fc(output)
        return output


class TransformerAttentionwithCatPe(nn.Module):
    def __init__(self, input_dim, output_dim, last_dim=1, token_num=2):
        super(TransformerAttentionwithCatPe, self).__init__()
        input_dim = input_dim * 2
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim * output_dim, last_dim)
        self.softmax = nn.Softmax(dim=1)
        self.position_embeddings = nn.Parameter(torch.zeros([token_num, input_dim//2]))

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.position_embeddings.expand(x.shape[0], -1, -1)
        x_with_position = torch.cat([x, pe], dim=-1)

        q = self.query(x_with_position)
        k = self.key(x_with_position)
        v = self.value(x_with_position)

        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))

        attention = self.softmax(attention)
        output = torch.matmul(attention, v)

        output = output.view([output.shape[0], -1])
        output = self.fc(output)
        return output

