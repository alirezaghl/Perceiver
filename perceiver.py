import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
from dataclasses import dataclass
from typing import Optional, Tuple
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from functools import wraps


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    print(f"Using device: {device}")

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

@dataclass
class PerceiverConfig:
    num_latents: int = 128
    num_z_channels: int = 256       
    qk_channels: Optional[int] = 256  
    v_channels: Optional[int] = 256   
    
    num_self_attends_per_block: int = 1    
    num_blocks: int = 2                     
    num_cross_attend_heads: int = 1        
    num_self_attend_heads: int = 8          
    
    cross_attend_widening_factor: int = 4   
    self_attend_widening_factor: int = 4    
    
    weight_tie_layers: bool = True          
    
    dropout_prob: float = 0.2               
    dropout_attn_prob: float = 0.2          
    
    batch_size: int = 128                    
    learning_rate: float = 1e-3             
    weight_decay: float = 0.05               
    num_epochs: int = 100
    
    num_classes: int = 10  
    input_channels: int = 3
    z_pos_enc_init_scale: float = 0.02
    cross_attention_shape_for_attn: str = 'kv'
    use_query_residual: bool = True

def attend(q, k, v, dropout_prob=0.0, attention_mask=None):
    batch, q_indices, num_heads, q_head_dim = q.shape
    _, _, _, v_head_dim = v.shape

    attention = torch.einsum('bthd,bThd->bhtT', q, k)
    scale = 1. / math.sqrt(q_head_dim)
    attention *= scale

    if attention_mask is not None:
        large_k = 1e4 if attention.dtype == torch.float16 else 1e30
        attention = attention.masked_fill(~attention_mask[:, None, :, :], -large_k)

    normalized = torch.softmax(attention, dim=-1)
    if dropout_prob > 0.0:
        normalized = F.dropout(normalized, dropout_prob)

    summed = torch.einsum('bhtT,bThd->bthd', normalized, v)
    summed = rearrange(summed, 'b t h d -> b t (h d)')
    return summed

class Conv1D(nn.Module):
    def __init__(self, input_channels, output_channels, init_scale=1.0, bias=True):
        super().__init__()
        self.linear = nn.Linear(input_channels, output_channels, bias=bias)
        std = math.sqrt(2.0 / (input_channels + output_channels)) * init_scale
        nn.init.normal_(self.linear.weight, mean=0.0, std=std)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = None
        self.normalized_shape = normalized_shape

    def forward(self, x):
        if self.norm is None:
            self.norm = nn.LayerNorm(self.normalized_shape).to(x.device)
        return self.norm(x)

def make_cross_attention_mask(query_mask, kv_mask):
    batch_size, query_len = query_mask.shape
    _, key_len = kv_mask.shape
    mask = torch.einsum('bq,bk->bqk', query_mask, kv_mask)
    assert mask.shape == (batch_size, query_len, key_len)
    return mask

class Attention(nn.Module):
    def __init__(
        self,
        num_heads=8,
        init_scale=1.0,
        with_final_bias=True,
        final_init_scale_multiplier=1.,
        dropout_prob=0.0,
        qk_channels=None,
        v_channels=None,
        output_channels=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.init_scale = init_scale
        self.with_final_bias = with_final_bias
        self.final_init_scale = final_init_scale_multiplier * init_scale
        self.dropout_prob = dropout_prob
        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.output_channels = output_channels

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.output_proj = None

    def forward(self, inputs_q, inputs_kv, attention_mask=None):
        if self.qk_channels is None:
            self.qk_channels = inputs_q.shape[-1]
        if self.v_channels is None:
            self.v_channels = self.qk_channels
        if self.output_channels is None:
            self.output_channels = self.v_channels

        if self.q_proj is None:
            if self.qk_channels % self.num_heads != 0:
                raise ValueError(f'qk_channels ({self.qk_channels}) must be divisible by num_heads ({self.num_heads}).')
            if self.v_channels % self.num_heads != 0:
                raise ValueError(f'v_channels ({self.v_channels}) must be divisible by num_heads ({self.num_heads}).')

            self.q_proj = Conv1D(inputs_q.shape[-1], self.qk_channels, self.init_scale).to(inputs_q.device)
            self.k_proj = Conv1D(inputs_kv.shape[-1], self.qk_channels, self.init_scale).to(inputs_q.device)
            self.v_proj = Conv1D(inputs_kv.shape[-1], self.v_channels, self.init_scale).to(inputs_q.device)
            self.output_proj = Conv1D(self.v_channels, self.output_channels, self.final_init_scale, bias=self.with_final_bias).to(inputs_q.device)

        q = self.q_proj(inputs_q)
        k = self.k_proj(inputs_kv)
        v = self.v_proj(inputs_kv)

        qk_channels_per_head = self.qk_channels // self.num_heads
        v_channels_per_head = self.v_channels // self.num_heads

        q = q.view(q.shape[0], q.shape[1], self.num_heads, qk_channels_per_head)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, qk_channels_per_head)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, v_channels_per_head)

        attended = attend(q, k, v, dropout_prob=self.dropout_prob, attention_mask=attention_mask)
        return self.output_proj(attended)

class MLP(nn.Module):
    def __init__(self, widening_factor=4, dropout_prob=0.0, init_scale=1.):
        super().__init__()
        self.widening_factor = widening_factor
        self.dropout_prob = dropout_prob
        self.init_scale = init_scale
        self.fc1 = None
        self.fc2 = None

    def forward(self, x, is_training=True):
        if self.fc1 is None:
            output_channels = x.shape[-1]
            self.fc1 = Conv1D(output_channels, self.widening_factor * output_channels, self.init_scale).to(x.device)
            self.fc2 = Conv1D(self.widening_factor * output_channels, output_channels, self.init_scale).to(x.device)

        dropout_prob = self.dropout_prob if is_training else 0.0
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return F.dropout(x, p=dropout_prob, training=is_training)

class SelfAttention(nn.Module):
    def __init__(
        self,
        widening_factor=PerceiverConfig.self_attend_widening_factor,
        dropout_prob=PerceiverConfig.dropout_prob,
        dropout_attn_prob=PerceiverConfig.dropout_attn_prob,
        num_heads=PerceiverConfig.num_self_attend_heads,
        att_init_scale=1.0,
        dense_init_scale=1.0,
        qk_channels=None,
        v_channels=None
    ):
        super().__init__()
        self.widening_factor = widening_factor
        self.dropout_prob = dropout_prob
        self.dropout_attn_prob = dropout_attn_prob
        self.num_heads = num_heads
        self.att_init_scale = att_init_scale
        self.dense_init_scale = dense_init_scale
        self.qk_channels = qk_channels
        self.v_channels = v_channels

        self.norm1 = None
        self.norm2 = None
        self.attention = None
        self.mlp = None

    def forward(self, inputs, attention_mask=None, is_training=True):
        if self.norm1 is None:
            dim = inputs.shape[-1]
            self.norm1 = LayerNorm(dim)
            self.norm2 = LayerNorm(dim)
            self.attention = Attention(
                num_heads=self.num_heads,
                init_scale=self.att_init_scale,
                qk_channels=self.qk_channels,
                v_channels=self.v_channels,
                dropout_prob=self.dropout_attn_prob
            )
            self.mlp = MLP(
                widening_factor=self.widening_factor,
                dropout_prob=self.dropout_prob,
                init_scale=self.dense_init_scale
            )

        dropout_prob = self.dropout_prob if is_training else 0.0

        x = inputs
        qkv_inputs = self.norm1(inputs)
        attention_out = self.attention(qkv_inputs, qkv_inputs, attention_mask)
        attention_out = F.dropout(attention_out, p=dropout_prob, training=is_training)
        x = x + attention_out
        x = x + self.mlp(self.norm2(x), is_training=is_training)
        return x

class CrossAttention(nn.Module):
    def __init__(
        self,
        widening_factor=PerceiverConfig.cross_attend_widening_factor,
        dropout_prob=PerceiverConfig.dropout_prob,
        dropout_attn_prob=PerceiverConfig.dropout_attn_prob,
        num_heads=PerceiverConfig.num_cross_attend_heads,
        att_init_scale=1.0,
        dense_init_scale=1.0,
        shape_for_attn='kv',
        use_query_residual=PerceiverConfig.use_query_residual,
        qk_channels=None,
        v_channels=None
    ):
        super().__init__()
        self.widening_factor = widening_factor
        self.dropout_prob = dropout_prob
        self.dropout_attn_prob = dropout_attn_prob
        self.num_heads = num_heads
        self.att_init_scale = att_init_scale
        self.dense_init_scale = dense_init_scale
        self.shape_for_attn = shape_for_attn
        self.use_query_residual = use_query_residual
        self.qk_channels = qk_channels
        self.v_channels = v_channels

        self.q_norm = None
        self.kv_norm = None
        self.attention = None
        self.mlp = None

    def forward(self, inputs_q, inputs_kv, attention_mask=None, is_training=True):
        if self.q_norm is None:
            self.q_norm = LayerNorm(inputs_q.shape[-1])
            self.kv_norm = LayerNorm(inputs_kv.shape[-1])

            if self.shape_for_attn == 'q':
                qk_channels = inputs_q.shape[-1]
            elif self.shape_for_attn == 'kv':
                qk_channels = inputs_kv.shape[-1]
            else:
                raise ValueError(f'Unknown value {self.shape_for_attn} for shape_for_attention.')

            if self.qk_channels is not None:
                qk_channels = self.qk_channels
            v_channels = self.v_channels

            self.attention = Attention(
                num_heads=self.num_heads,
                init_scale=self.att_init_scale,
                dropout_prob=self.dropout_attn_prob,
                qk_channels=qk_channels,
                v_channels=v_channels,
                output_channels=inputs_q.shape[-1]
            )

            self.mlp = MLP(
                widening_factor=self.widening_factor,
                dropout_prob=self.dropout_prob,
                init_scale=self.dense_init_scale
            )

        dropout_prob = self.dropout_prob if is_training else 0.0
        q_normalized = self.q_norm(inputs_q)
        kv_normalized = self.kv_norm(inputs_kv)
        attention_out = self.attention(q_normalized, kv_normalized, attention_mask=attention_mask)
        attention_out = F.dropout(attention_out, p=dropout_prob, training=is_training)
        if self.use_query_residual:
            x = inputs_q + attention_out
        else:
            x = attention_out
        x = x + self.mlp(self.q_norm(x), is_training=is_training)
        return x

class TrainablePositionEncoding(nn.Module):
    def __init__(self, index_dim, num_channels=PerceiverConfig.num_z_channels, init_scale=0.02):
        super().__init__()
        self.index_dim = index_dim
        self.num_channels = num_channels
        self.pos_embs = nn.Parameter(torch.zeros(index_dim, num_channels))
        nn.init.trunc_normal_(self.pos_embs, std=init_scale)

    def forward(self, batch_size, pos=None):
        if batch_size is not None:
            return self.pos_embs.unsqueeze(0).expand(batch_size, -1, -1)
        return self.pos_embs

class PerceiverEncoder(nn.Module):
    def __init__(self, config: PerceiverConfig):
        super().__init__()
        self.config = config
        
        if config.num_z_channels % config.num_self_attend_heads != 0:
            raise ValueError(f'num_z_channels must be divisible by num_self_attend_heads')
        if config.num_z_channels % config.num_cross_attend_heads != 0:
            raise ValueError(f'num_z_channels must be divisible by num_cross_attend_heads')

        self.z_pos_enc = TrainablePositionEncoding(
            index_dim=config.num_latents,
            num_channels=config.num_z_channels,
            init_scale=config.z_pos_enc_init_scale
        )

        get_cross_attn = cache_fn(lambda: CrossAttention(
            num_heads=config.num_cross_attend_heads,
            widening_factor=config.cross_attend_widening_factor,
            shape_for_attn=config.cross_attention_shape_for_attn,
            qk_channels=config.qk_channels,
            v_channels=config.v_channels,
            use_query_residual=config.use_query_residual,
            dropout_prob=config.dropout_prob,
            dropout_attn_prob=config.dropout_attn_prob
        ))

        get_self_attn = cache_fn(lambda: SelfAttention(
            num_heads=config.num_self_attend_heads,
            widening_factor=config.self_attend_widening_factor,
            qk_channels=config.qk_channels,
            v_channels=config.v_channels,
            dropout_prob=config.dropout_prob,
            dropout_attn_prob=config.dropout_attn_prob
        ))

        # weight sharing
        self.blocks = nn.ModuleList([])
        for i in range(config.num_blocks):
            should_cache = i > 0 and config.weight_tie_layers
            cache_args = {'_cache': should_cache}

            cross_attn = get_cross_attn(**cache_args, key=f'cross_{i}')
            self_attns = nn.ModuleList([
                get_self_attn(**cache_args, key=f'self_{i}_{j}')
                for j in range(config.num_self_attends_per_block)
            ])
            
            self.blocks.append(nn.ModuleList([cross_attn, self_attns]))

    def latents(self, inputs):
        return self.z_pos_enc(batch_size=inputs.shape[0])

    def forward(self, inputs, z, is_training=True, input_mask=None):
        attention_mask = None
        if input_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=torch.ones(z.shape[:2], device=z.device, dtype=torch.bool),
                kv_mask=input_mask
            )

        for cross_attn, self_attns in self.blocks:
            # Cross attention
            z = cross_attn(z, inputs, is_training=is_training, attention_mask=attention_mask)
            
            # Self attention blocks
            for self_attn in self_attns:
                z = self_attn(z, is_training=is_training)

        return z


class BasicDecoder(nn.Module):
    def __init__(
        self,
        output_index_dims,
        output_num_channels,
        num_z_channels=PerceiverConfig.num_z_channels,
        num_heads=PerceiverConfig.num_cross_attend_heads,
        qk_channels=None,
        v_channels=None,
        use_query_residual=False,
        position_encoding_type='trainable'
    ):
        super().__init__()
        self.output_index_dims = output_index_dims
        self.output_num_channels = output_num_channels
        self.num_z_channels = num_z_channels

        self.output_pos_enc = TrainablePositionEncoding(
            index_dim=np.prod(output_index_dims),
            num_channels=num_z_channels
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            qk_channels=qk_channels,
            v_channels=v_channels,
            use_query_residual=use_query_residual
        )

        self.final_layer = None

    def decoder_query(self, inputs):
        return self.output_pos_enc(inputs.shape[0])

    def forward(self, query, z, is_training=True):
        if self.final_layer is None:
            self.final_layer = Conv1D(self.num_z_channels, self.output_num_channels).to(query.device)
        attended = self.cross_attention(query, z)
        return self.final_layer(attended)

class ClassificationDecoder(nn.Module):
    def __init__(
        self,
        num_classes,
        num_z_channels=PerceiverConfig.num_z_channels,
        num_heads=PerceiverConfig.num_self_attend_heads
    ):
        super().__init__()
        self.num_classes = num_classes

        self.decoder = BasicDecoder(
            output_index_dims=(1,),
            output_num_channels=num_classes,
            num_z_channels=num_z_channels,
            num_heads=num_heads
        )

    def decoder_query(self, inputs):
        return self.decoder.decoder_query(inputs)

    def forward(self, query, z, is_training=True):
        logits = self.decoder(query, z, is_training)
        return logits[:, 0, :]

class Perceiver(nn.Module):
    def __init__(self, config: PerceiverConfig):
        super().__init__()
        self.encoder = PerceiverEncoder(config)
        self.decoder = ClassificationDecoder(
            num_classes=config.num_classes,
            num_z_channels=config.num_z_channels,
            num_heads=config.num_self_attend_heads
        )

        self.input_pos_enc = TrainablePositionEncoding(
            index_dim=32*32,
            num_channels=config.num_z_channels,
            init_scale=0.02
        )

        self.input_embedding = nn.Linear(3, config.num_z_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, h * w, c)  
        x = self.input_embedding(x) 
        pos = self.input_pos_enc(b)
        x = x + pos
        latents = self.encoder.latents(x)   
        z = self.encoder(x, latents)
        decoder_query = self.decoder.decoder_query(x)
        logits = self.decoder(decoder_query, z)
        return logits


def train_cifar(config: PerceiverConfig):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    model = Perceiver(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)  

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs) 
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f'Epoch {epoch + 1}: Accuracy: {100.*correct/total:.2f}%')

if __name__ == '__main__':
    config = PerceiverConfig()
    
    model = Perceiver(config).to(device)
    model = torch.compile(model, mode='reduce-overhead')
    train_cifar(config)