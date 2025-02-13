# Perceiver Implementation for Image Classification

This repository contains a PyTorch implementation of the Perceiver architecture for image classification tasks. The Perceiver is a general-purpose architecture that uses cross-attention to process arbitrary input data through a bottleneck of learned latent vectors.

![model](https://github.com/alirezaghl/Perceiver/blob/main/figs/model.png)

## Configuration


```python
@dataclass
class PerceiverConfig:
    num_latents: int = 128            # Number of latent vectors
    num_z_channels: int = 256         # Dimension of latent vectors
    num_blocks: int = 2               # Number of encoder blocks
    num_cross_attend_heads: int = 1   # Heads in cross-attention
    num_self_attend_heads: int = 8    # Heads in self-attention
    learning_rate: float = 1e-3       # Initial learning rate
    # ... other parameters
```

## Training

To train the model:

```python
python perceiver.py
```


## References

1. Perceiver architecture from "Perceiver: General Perception with Iterative Attention" (Jaegle et al., 2021)
2. Original implementation: [DeepMind Research Repository](https://github.com/google-deepmind/deepmind-research/tree/master)
