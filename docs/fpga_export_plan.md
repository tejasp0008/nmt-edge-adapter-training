# FPGA Export Plan (generated)

## File formats

### .wbin (weights)
- Header (16 bytes, little endian):
  - magic: 4 bytes ASCII 'EAQ1'
  - version: uint32
  - out_dim: uint32
  - in_dim: uint32
- scales: out_dim values as FP16 (2 bytes each)
- packed weights: ceil(in_dim/2) * out_dim bytes (two 4-bit signed values per byte; low nibble = col0, high nibble = col1), row-major
- file padded to 64-byte boundary

### adapter_<lang>.bin
- (example binary layout)
- header: magic 'ADAP', version u32, num_layers u32
- per-layer index: layer_name_len u16, layer_name bytes, rank u16, offset u64, size u64
- payload: concatenated A (r x in fp16), followed by B (out x r fp16)

### Embeddings
- core_embeddings.bin: FP16 contiguous (core_size x emb_dim)
- shard files: emb_shard_<lang>.bin: FP16 contiguous (num_tokens x emb_dim)

## Memory map (suggested)
- HBM: store .wbin weights
- DDR: adapter bins, embedding shards
- URAM/BRAM: universal core embeddings + active adapter slots (double-buffer)

## Alignment
- All files must be 64-byte aligned for DMA efficiency.

## CSR register map (example)
- 0x0010 REG_ADAPTER_ACTIVE_PTR_LO (u32)
- 0x0014 REG_ADAPTER_ACTIVE_PTR_HI (u32)
- 0x0018 REG_ADAPTER_PREFETCH_PTR_LO (u32)
- 0x001C REG_ADAPTER_PREFETCH_PTR_HI (u32)
- 0x0020 REG_ADAPTER_CMD (u32)  1=prefetch_start 2=swap_active
- 0x0024 REG_PREFETCH_STATUS (u32): bit0=prefetch_busy bit1=prefetch_done

## Notes
- Endianness: little-endian throughout.
- If scales require FP32 instead of FP16, update header/spec and unpacker accordingly.

