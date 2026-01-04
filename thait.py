# Thai Transformer backbone
import math
import torch
from torch import Tensor
from torch import nn

# --------------------------------------------------------------------------------------------------------------------------- #
# Activation Functions

class GELUActivation(nn.Module):
    """
    Standard GELU (tanh approximation), commonly used in BERT/GPT models.
    Reference: Hendrycks & Gimpel (2016)
    """
    def forward(self, input):
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
    
class NewGELUActivation(nn.Module):
    """
    New GELU activation function (sigmoid-based approximation) used in ThaiT FFN.
    """
    def forward(self, input: Tensor) -> Tensor:
        # Sigmoid-based GELU approximation
        # NewGELU(x) = x * sigmoid(1.702 * x)
        return input * torch.sigmoid(1.702 * input)
    
# --------------------------------------------------------------------------------------------------------------------------- #
# Patch Embedding + Token Embeddings

class PatchEmbeddings(nn.Module):
    """
    Convert an image into a sequence of patch embeddings.
    Uses Conv2d with kernel_size=patch_size and stride=patch_size to create non-overlapping patches.
    """
    def __init__(self, config):
        super().__init__()
        self.image_size   = config["image_size"]
        self.patch_size   = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size  = config["hidden_size"]

        # Number of patches = (H/P) * (W/P)
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Conv projection:
        # Input: (B, C, H, W) -> Output: (B, hidden_size, H/P, W/P)
        self.projection = nn.Conv2d(
            self.num_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
    def forward(self, x):
        # Project patches + Flatten spatial grid into sequence
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class Embeddings(nn.Module):
    """
    Create the full ThaiT token sequence: [CLS] + Patch Embeddings + Position Embeddings
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"])
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forword(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()

        # Expand [CLS] token to batch size:
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x
    
# --------------------------------------------------------------------------------------------------------------------------- #
# Attention Modules (standard multi-head self-attention: MSA)
class AttentionHead(nn.Module):
    """
    Single self-attention head:
    Computes Q, K, V and outputs:
        - attention_output: (B, N, head_dim)
        - attention_probs : (B, L, L)
    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key   = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
        # Dropout layer applied to attention probabilities
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, L, D)
        query = self.query(x)
        key   = self.key(x)
        value = self.value(x)
        # Attention scores: QK^T / sqrt(d_k)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        return attention_output, attention_probs
    
class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention (MSA) module:
    - Builds H separate heads
    - Concatenates outputs and projects back to hidden_size
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]

        # head_dim = D / H
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        # Output projection: concat heads -> hidden_size
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat(
            [attention_output for attention_output, _ in attention_outputs], 
            dim=-1
        )
        attention_output = self.output_projection(attention_output)
        attention_output = self.dropout(attention_output)
        if not output_attentions:
            return (attention_output, None)
        else:
            # Stack head attention probs: -> (B, H, L, L)
            attention_probs = torch.stack(
                [attention_probs for _, attention_probs in attention_outputs], 
                dim=1)
            return (attention_output, attention_probs)

# --------------------------------------------------------------------------------------------------------------------------- #
# Optimizing MHA (Optimizing Multi-Head Attention) with fused QKV projection used in ThaiT
class OptimizingMultiHeadAttention(nn.Module):
    """
    Fused QKV multi-head attention:
    - Uses a single linear layer to compute [Q, K, V] simultaneously.
    - Improves GPU locality and reduces overhead (ThaiT optimization).]
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]

        # One projection to produce Q, K, V together:
        # Input: (B, L, D) -> Output: (B, L, 3*D)
        self.qkv_projection = nn.Linear(
            self.hidden_size,
            self.all_head_size * 3,
            bias=self.qkv_bias
        )
        self.attn.dropout = nn.Dropout(config["attention_probs_dropout_prob"])

        # Output projection back to D
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # qkv: (B, L, 3*D)
        qkv = self.qkv_projection(x)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key   = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Attention scores: (B, H, L, L)
        attention_score = torch.matmul(query, key.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_score, dim=-1)
        attention_probs = self.attn.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)

        # Merge heads + Final projection
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.all_head_size)
        )
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)

# --------------------------------------------------------------------------------------------------------------------------- #
# Feed-forward network (FFN / MLP inside Transformer block) used in ThaiT
class MLP(nn.Module):
    """
    Transformer FFn block:
    Dense -> Activation -> Dense -> Dropout
    """
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x
    
# --------------------------------------------------------------------------------------------------------------------------- #
# Transformer Block + Encoder
class Block(nn.Module):
    """
    One Transformer encoder block:
    LayerNorm -> MHA -> Residual 
    LayerNorm -> FFN -> Residual
    """
    def __init__(self, config):
        super().__init__()

        # Switch between standard MHA and Optimizing MHA
        self.use_optimizing_attention = config.get("use_optimizing_attention", False)
        if self.use_optimizing_attention:
            self.attention = OptimizingMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])
    
    def forward(self, x, output_attentions=False):
        attention_output, attention_probs = self.attention(
            self.layernorm_1(x),
            output_attentions=output_attentions
        )
        x = x + attention_output
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)

class Encoder(nn.Module):
    """
    Stack of Transformer blocks.
    """
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            self.blocks.append(Block(config))

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)

        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)
        
# --------------------------------------------------------------------------------------------------------------------------- #
# ThaiT for classification
class ThaiTForClassification(nn.Module):
    """
    ThaiT-style classifier:
    Image -> Embeddings -> Encoder -> task [CLS] -> Linear classifier
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.apply(self._init_weights)
    
    def forward(self, x, output_attentions=False):
        # Tokenize + add embeddings
        embedding_output = self.embeddings(x)
        # Encoder tokens through Transformer blocks
        encoder_output, all_attentions = self.encoder(
            embedding_output, output_attentions=output_attentions
        )
        # Use [CLS] token output for classification
        logits = self.classifier(encoder_output[:, 0, :])
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
    def _init_weights(self, module):
        # Initialize Linear and Conv weights with normal distribution
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config["initializer_range"]
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # LayerNorm initialization
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)

# --------------------------------------------------------------------------------------------------------------------------- #
