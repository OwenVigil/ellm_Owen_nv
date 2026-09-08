## Decode kernel fusion experiment
* idea: Decoding is a memory-bound operation, and by default all context's kv caches are fully stored, demanding tremendous HBM storage capacity, which also reduces the query batches the server can handle.

* But if we drop some kv caches optionally, we releases the HBM. So we can handle a larger query batch at the same time, and make more use of the compute unit like tensor-core or ALU when running the memory-bound decoding process.

## System design
* based hardware platform: 4 * A100(80GB), CUDA:13.1
* based library： vllm v0.4.2(path: vllm/,switch to the branch manually. Using the low version is to simplify the coding in paged attention, rather than in flash attention)
* model: llama-13B
* burden: the requests are always long, and HBM is always in shortage. When HBM has margin, we can return keep all kv caches.
### Key fusion design:
* 1.Prefill the prompt. 
* 2.Firstly, drop all kv caches of the `drop_len` number of tokens in the first part of the sequence. But keep the embedding sequence.
* 3.`kernel_a` ： # recompute the kv caches of the `drop_len` number of tokens in the first part of the sequence from previous layer's hidden_states
  * Input: embedding sequence from layer 0/ hidden_states from last layer.
  * output: recomputed kv caches of the `drop_len` number of tokens in the first part of the sequence.
* 4.`kernel_b` ： # decoding the layer with all kv caches include recomputed and kept kv caches.
  * Input: recomputed kv caches, kept kv caches
  * output: probility of the next token, `drop_len` number of tokens' hidden_states of this layer for next layer's kv caches recomputation.
* 5.After decoding the layer, drop the recomputed kv caches and hidden_states of the `drop_len` number of tokens in the first part of the sequence

#### For example:
* 1. We have a sequence of 100 tokens, and we set `drop_len`=32. And the model has 40 layers.
* 2. We prefill, and get the `embedding sequence` and kv caches of the 100 tokens.
* 3. We drop the kv caches of the first 32 tokens of the 40 layers. So we dropped 2*32*40*hidden_dim*DataType_size bytes of HBM.
* 4. We recompute the kv caches of the first 32 tokens of the layer 0 from the embedding sequence, and then decode the layer 0 with all kv caches include recomputed and kept kv caches. After decoding, we drop the recomputed kv caches. Then we have the next token's probability and hidden_states of the first 32 tokens of layer 1.
* 5. Similarly, we recompute the kv caches of the first 32 tokens of layer 1 from the hidden_states of layer 1, and then decode the layer 1 with all kv caches include recomputed and kept kv caches. After decoding, we drop the recomputed kv caches. Then we have the next token's probability and hidden_states of the first 32 tokens of layer 2. But different from layer 0, we also drop the hidden_states of the first 32 tokens of layer 1.
* 6. But what's important is that in fact, decoding and recomputing kv caches are fused, so we can handle the two processes at the same time. That's why we can balance the memory-bound and compute-bound operations.
##### graph like:
```
sequence:  [token_0, token_1, token_2, ..., token_99]

embedding sequence: [embedding_0, embedding_1, embedding_2, ..., embedding_99]

layer 0: [dropped, dropped, dropped, ..., kept_kv_cache_32, kept_kv_cache_33, ..., kept_kv_cache_99]

{kernel_a: recompute kv caches of the first 32 tokens from embedding sequence}
{kernel_b: decode the layer with all kv caches include recomputed and kept kv caches}
{The recomputed kv caches of the first 32 tokens are dropped after decoding}

layer 1: [dropped, dropped, dropped, ..., kept_kv_cache_32, kept_kv_cache_33, ..., kept_kv_cache_99]

{kernel_a: recompute kv caches of the first 32 tokens from hidden_states computed from layer 0}
{kernel_b: decode the layer with all kv caches include recomputed and kept kv caches}
{The recomputed kv caches of the first 32 tokens and hidden_states are dropped after decoding}

layer 2: [dropped, dropped, dropped, ..., kept_kv_cache_32, kept_kv_cache_33, ..., kept_kv_cache_99]

...

layer 39: [dropped, dropped, dropped, ..., kept_kv_cache_32, kept_kv_cache_33, ..., kept_kv_cache_99]
```

#### one branch point
* Fusion way 1:
`kernel_a` recomputes the kv cache for the layer.
`kernel_b` decodes self-attention and feed-forward for the layer, fistly using the kept kv caches and then using the recomputed kv caches.
The benefit of this way is that it fits the 40 layers, no special design for the first layer. 

* Fusion way 2:
`kernel_a` recomputes the kv cache for the next layer.
`kernel_b` decodes self-attention and feed-forward for the layer, using the kept kv caches and the recomputed kv caches.
But this way needs a special design for the first layer, for no kernel_a rebuilding kv cache for it.

### stream design
There are two streams:
one for recomputing, as designed above, S_rec.
one for swapping the discarded kv caches from the main memory to HBM, S_swap.


### Extra adjust parameter design
see: [supplement.md](supplement.md)