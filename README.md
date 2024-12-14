# 🐍 Jax Mamba: Minimal State Space Model Implementation

## About

This repository contains a lightweight, fast implementation of the Mamba (Selective State Space) model using JAX and Flax. Inspired by the groundbreaking paper "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" by Albert Gu and Tri Dao, this project provides an accessible and understandable implementation of state-of-the-art sequence modeling technology.

### Key Features

- ⚡ Minimal and efficient JAX implementation
- 🧠 Supports pretrained Mamba models from HuggingFace
- 🔬 Saved and added detailed comments explaining ssm mechanisms
- 🚀 JAX-powered for high-performance computing
- 📝 Easy model loading and text generation
- 💫 **O(log n)** jax parallel scan complexity
- ⚡️ very fast with padding and fixed input lenght

The Mamba model introduces a novel approach to sequence modeling by:

- Using **selective state spaces** that adaptively remember or forget information
- Achieving **linear-time complexity** in sequence length
- Providing an alternative to traditional transformer architectures


## Installation

```bash
# Clone the repository
git clone https://github.com/Vadimbuildercxx/jax-mamba.git
cd jax-mamba

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Loading a Pretrained Model

```python
from model import Mamba

# Load a pretrained Mamba model
model, params = Mamba.from_pretrained('state-spaces/mamba-370m')
```

### Text Generation Example

```python
import jax
from utils import generate
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
out = generate(
    model, 
    key=jax.random.PRNGKey(42), 
    params=params, 
    tokenizer=tokenizer, 
    prompt='Mamba is the', 
    n_tokens_to_gen=40
)
print(out)
```
🤔 The unknown first game on the Nintendo switch?
```
'Mamba is the first game to be released on the Nintendo Switch. It is a side-scrolling platformer that is set in a futuristic world where the player must fight against the evil forces of the Mamba'
```

## Technical Overview

The Mamba model introduces a novel approach to sequence modeling by:

- Using **selective state spaces** that adaptively remember or forget information
- Achieving **linear-time complexity** in sequence length
- Providing an alternative to traditional transformer architectures

### Key Components

- `MambaBlock`: Core selective state space mechanism
- `ResidualBlock`: Residual connections and normalization
- `RMSNorm`: Root Mean Square Layer Normalization
- Efficient discretization of state space parameters

## References

1. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
2. [The Annotated S4](https://srush.github.io/annotated-s4)
3. [Mamba: The Hard Way](https://srush.github.io/annotated-mamba/hard.html)
4. [Visual Guide to Mamba]https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Acknowledgements

- Albert Gu and Tri Dao for the original Mamba paper
- Sasha Rush for annotated implementations
- The JAX and Flax communities
