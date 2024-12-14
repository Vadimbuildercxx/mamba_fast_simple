import flax.nnx
import jax.numpy as jnp
import jax 
import flax
import time

def left_pad_tensor(tensor, max_tokens, padding_value=0):
    """
    Left pads a tensor to the nearest power of two along the second dimension.
    
    Args:
        tensor (jax.numpy.ndarray): Tensor of shape (batch_size, seq_len).
        padding_value (int): Value used for padding (e.g., pad_token_id).
        
    Returns:
        jax.numpy.ndarray: Padded tensor to nearest power of two.
    """
    batch_size, max_seq_len = tensor.shape
    # Find the nearest power of two greater than or equal to max_seq_len
    pad_size = max_seq_len + max_tokens

    # Calculate padding needed on the left side
    pad_width = [(0, 0), (pad_size - max_seq_len, 0)]  # Padding on the second dimension (seq_len)
    
    # Apply left padding to the tensor
    padded_tensor = jnp.pad(tensor, pad_width, mode="constant", constant_values=padding_value)
    
    return padded_tensor


def generate(
        model: flax.nnx.Module,
        key,
        params,
        tokenizer,
        prompt: str,
        sample:bool = True,
        top_k: int = 40,
        n_tokens_to_gen: int = 50,
        pad = True,
        pad_token_id: int = 1,
        do_jit: bool = True,
        deterministic: bool = True):
    
    # JIT-compile the inference function
    def inference_model(params, x):
        return model.apply(params, x)
    
    def inference_step(key, next_token_logits):
        probs = jax.nn.softmax(next_token_logits, axis=-1)
        (batch, vocab_size) = probs.shape
        
        if top_k is not None:
            (values, indices) = jax.lax.top_k(probs, k=top_k)
            probs = jnp.where(probs < values[:, -1, None], 0, probs)
            probs = probs / probs.sum(axis=1, keepdims=True)
        
        if sample:
            if not deterministic:
                current_time = int(time.time())
                key = jax.random.PRNGKey(current_time)
            next_indices = jax.random.categorical(key, jnp.log(probs), shape=(1,))
        else:
            next_indices = jnp.argmax(probs, axis=-1)
        
        return next_indices
    
    if do_jit:
        inference_model_fn = jax.jit(inference_model)
        inference_step_fn = jax.jit(inference_step)
    else:
        inference_model_fn = inference_model
        inference_step_fn = inference_step

    if pad:
        input_ids = left_pad_tensor(jnp.array(tokenizer(prompt, return_tensors='np').input_ids), n_tokens_to_gen, pad_token_id)
    else: 
        input_ids = jnp.array(tokenizer(prompt, return_tensors='np').input_ids)
    
    for token_n in range(n_tokens_to_gen):
        indices_to_input = input_ids

        next_token_logits = inference_model_fn(params, jax.lax.stop_gradient(indices_to_input))[:, -1]
        next_indices = inference_step_fn(key, next_token_logits)
        if pad:
            input_ids = jnp.concat([input_ids[:,1:], next_indices[:, jnp.newaxis]], axis=1)
        else:
            input_ids = jnp.concat([input_ids, next_indices[:, jnp.newaxis]], axis=1)

    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]
    
    return output_completions