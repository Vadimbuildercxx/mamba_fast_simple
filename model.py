"""Simple, fast modified implementation of Mamba in PyTorch. 
   Inspired by John (Zhiyao) Ma awesome repo: https://github.com/johnma2006/mamba-minimal/

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4
    [3] Mamba: The Hard Way (Sasha Rush)
        https://srush.github.io/annotated-mamba/hard.html

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
from typing import Union

import flax
from flax import nnx
from flax.training import train_state, checkpoints
from flax.core import FrozenDict
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.profiler
from jax import lax

from dataclasses import dataclass
from einops import rearrange, repeat, einsum

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    K_size: int = 16
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    key = jax.random.PRNGKey(42)
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):

    args: ModelArgs

    def setup(self):
        """Full Mamba model."""
        #self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.embedding = nn.Embed(self.args.vocab_size, self.args.d_model)
        #self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.layers = [ResidualBlock(self.args) for _ in range(self.args.n_layer)]
        self.norm_f = RMSNorm(self.args.d_model)

        #self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head = nn.Dense(self.args.vocab_size, use_bias=True)

    def initialize_params(self, rng, input_shape):
        """
        Initializes parameters and ties embedding weights to lm_head weights.
        """
        params = self.init(rng, jnp.ones(input_shape))

        # Tie lm_head kernel to embedding weights
        # Tie output projection to embedding weights.
        # See "Weight Tying" paper
        params['params']['lm_head']['kernel'] = params['params']['embedding']['embedding']

        return params

    def __call__(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """

        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits

    
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        import torch
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)

        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        
        x = jnp.ones((2, 5), dtype=jnp.int32)
        params = model.init(args.key, x) # Initialization call

        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key].cpu().numpy()
        
        del state_dict

        params["params"]["embedding"]["embedding"] = jnp.array(new_state_dict["embedding.weight"].copy())
        for k, v in list(new_state_dict.items())[1:-2]:
            path = k.split(".")
            if path[0] == "layers":
                path = [path[0] + "_" + path[1]] + path[2:]

            total_path = ""
            curr = params["params"]
            for p in path[:-1]:
                
                curr = curr[p]
                total_path += p + "." 
            
            if p == "conv1d":
                p = "Conv_0"
                curr = curr[p]

            if path[-1] == "weight" and p != "norm":
                path[-1] = "kernel"

            if curr[path[-1]].shape == new_state_dict[k].T.shape:
                curr[path[-1]] = jnp.array(new_state_dict[k].T.copy())
            else:
                curr[path[-1]] = jnp.array(new_state_dict[k].copy())
            
        params["params"]["norm_f"]["weight"] = jnp.array(new_state_dict["norm_f.weight"].copy())
        params["params"]["lm_head"]["kernel"] = jnp.array(new_state_dict["lm_head.weight"].T.copy())

        return model, params

class ResidualBlock(nn.Module):
    args: ModelArgs

    def setup(self):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        self.mixer = MambaBlock(self.args)
        self.norm = RMSNorm(self.args.d_model)
        

    def __call__(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x)) + x

        return output
            

class DepthwiseConv1D(nn.Module):
    features: int
    kernel_size: int

    @nn.compact
    def __call__(self, x):
        return nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            feature_group_count=self.features,
            strides=(1,),
            padding=self.kernel_size - 1,
        )(x)


class MambaBlock(nn.Module):

    args: ModelArgs

    def setup(self):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        #self.in_proj = nn.Dense(args.d_model, args.d_inner * 2, use_bias=args.bias)
        self.in_proj = nn.Dense(self.args.d_inner * 2, use_bias=self.args.bias)

        self.conv1d = DepthwiseConv1D(
            features=self.args.d_inner, 
            kernel_size=self.args.d_conv
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C or s_B(x), s_C(x), s_Δ(x) in Mamba paper
        #self.x_proj = nn.Dense(args.d_inner, args.dt_rank + args.d_state * 2, use_bias=False)
        self.x_proj = nn.Dense(self.args.dt_rank + self.args.d_state * 2, use_bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in  or tau_Δ(x) in Mamba paper
        #self.dt_proj = nn.Dense(args.dt_rank, args.d_inner, use_bias=True)
        self.dt_proj = nn.Dense(self.args.d_inner, use_bias=True)

        self.A_log = self.param("A_log", self.a_log_initializer, (self.args.d_state, self.args.d_inner))
        self.D = self.param("D", self.d_initializer, (self.args.d_inner,))
        #self.out_proj = nn.Dense(args.d_inner, args.d_model, use_bias=args.bias)
        self.out_proj = nn.Dense(self.args.d_model, use_bias=self.args.bias)
    
    def d_initializer(self, rng, shape, dtype=jnp.float32):
        # Custom initialization function that returns ones
        return jnp.ones(shape, dtype=dtype)
    
    def a_log_initializer(self, rng, shape):
        # Custom initialization function that returns ones
        return jnp.log( repeat(jnp.arange(1, shape[0]  + 1), 'n -> d n', d=shape[1]))

    def __call__(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        
        (x, res) = jnp.split(x_and_res, [self.args.d_inner, ], axis=-1)

        x = self.conv1d(x)[:,:l,:]

        x = nn.activation.silu(x)

        y = self.ssm(x)

        y = y * nn.activation.silu(res)
        
        output = self.out_proj(y)

        return output
    
    def ssm(self, u):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        A = -jnp.exp(self.A_log.astype(float))  # shape (d_in, n)
        D = self.D.astype(float)
        
        x_dbl = self.x_proj(u)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = jnp.split(x_dbl, [self.args.dt_rank, self.args.dt_rank + n,], axis=-1)
        
        delta = nn.activation.softplus(self.dt_proj(delta))  # (b, l, d_in)

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = jnp.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in  -> b l d_in n')

        # This modification of run_SSM(A, B, C, u) in The Annotated S4 [2] and Mamba: The Hard Way[3] triton implementation
        y = MambaBlock.run_SSM(deltaA, deltaB_u, C) + u * D 

        return y

    @jax.jit
    def run_parallel_scan(Ab, Bb_u, Cb):
        # Associative operation inspired from "Annotated Mamba" by S.Rush
        def combine_parallel(state1, state2):
            # (a_1​,b_1​)⊕(a_2​,b_2​)=(a_1 * ​a_2​,a_2 *​ b_1​ + b_2​)
            fl, xl = state1
            fr, xr = state2
            f = fr * fl
            x = fr * xl + xr
            return f, x

        # Perform associative scan
        results = jax.lax.associative_scan(combine_parallel, (Ab, Bb_u))

        return einsum(results[1], Cb, 'l b d_in n, l b n -> l b d_in')

    @jax.jit
    def run_SSM(deltaA, deltaB_u, C):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_parallel_scan(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        # Run recurrence
        # Also you can see selective scan in Annotated S4 [2] 
        ys = MambaBlock.run_parallel_scan(deltaA.swapaxes(0, 1), deltaB_u.swapaxes(0, 1), C.swapaxes(0, 1))
        y = ys.swapaxes(0, 1)

        return y


class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-5

    def setup(self):
        self.weight = self.param("weight", self.init_weight)

    def init_weight(self, rng):
        return jnp.ones(self.d_model)

    def __call__(self, x):
        output = x * 1. / jnp.sqrt(jnp.power(x, 2).mean(-1, keepdims=True) + self.eps) * self.weight
        return output


if __name__ == "__main__":
    # Test run
    args = ModelArgs(d_model=768, n_layer=24, vocab_size=50280, d_state=16, expand=2, dt_rank=48, K_size=16, d_conv=4, pad_vocab_size_multiple=8, conv_bias=True, bias=False)
    
    model = Mamba(args=args)
    x = jnp.array([[0,1,4],[0,1,4],[0,1,4],[0,1,4]])
    params = model.init(args.key, x) # Initialization call
    
    x = model.apply(params, x)
    loss_grad_fn = jax.value_and_grad(jnp.mean)
    loss_val, grads = loss_grad_fn(x)
    print(loss_val, grads)