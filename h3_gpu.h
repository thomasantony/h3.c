#ifndef H3_GPU_H
#define H3_GPU_H

#include <stddef.h>
#include <stdint.h>

typedef struct h3_gpu h3_gpu;
typedef struct h3_gpu_tensor h3_gpu_tensor;

typedef enum {
    H3_GPU_F32 = 0,
    H3_GPU_BF16,
    H3_GPU_I8,
    H3_GPU_U32
} h3_gpu_dtype;

typedef struct {
    uint64_t allocated_bytes;
    uint64_t live_bytes;
    uint64_t peak_live_bytes;
    uint64_t tensor_allocations;
    uint64_t direct_dispatches;
    uint64_t mps_linear_dispatches;
    uint64_t mps_conv_dispatches;
    uint64_t mps_sdpa_dispatches;
    uint64_t blit_copies;
    uint64_t submissions;
    double command_encode_seconds;
    double command_wait_seconds;
    /* Root MTLCommandBuffer timestamps; MPSGraph may schedule child buffers,
     * so command_wait_seconds is the complete turnaround measurement. */
    double gpu_seconds;
} h3_gpu_stats;

h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size);
void h3_gpu_free(h3_gpu *gpu);
int h3_gpu_is_m5(const h3_gpu *gpu);
int h3_gpu_has_nax_mlp(const h3_gpu *gpu);
int h3_gpu_has_int8_mlp(const h3_gpu *gpu);

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements);
h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements);
h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements);
/* Allocate shared Metal storage and pread BF16 payload directly into it. */
h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements);
/* Fill an existing shared BF16 buffer from a file. The tensor and its
 * accounting are unchanged, so this may run on an I/O thread while another
 * tensor is in flight on the GPU. */
int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size);
/* As above, but ask Darwin to avoid retaining a second copy in the file cache.
 * Intended for large sequential weight streams whose destination is the only
 * useful resident copy. */
int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size);
void h3_gpu_tensor_free(h3_gpu_tensor *tensor);
size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor);
h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor);
int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements);
int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements);
int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements);
int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements);
int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements);
int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements);
int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements);

int h3_gpu_begin(h3_gpu *gpu);
/* Commit the current command buffer without waiting, then continue encoding on
 * the same ordered queue. h3_gpu_submit() waits and validates the whole chain. */
int h3_gpu_continue(h3_gpu *gpu);
int h3_gpu_submit(h3_gpu *gpu);
const char *h3_gpu_error(const h3_gpu *gpu);
int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats);
/* Optional benchmark labels. With H3_PROFILE set, marks and context teardown
 * print wall time alongside command-buffer GPU time and allocation counters. */
void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label);
void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase);

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim);
int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3_gpu_patch_linear_bf16_offset(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             size_t output_offset,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3_gpu_patch_linear_bf16_map(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias,
                             const h3_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements);
/* The destination/source retain 16-bit storage; these convert the bit encoding
 * for FP16 MPS kernels without allocating another activation arena. */
int h3_gpu_copy_bf16_fp16(h3_gpu *gpu, h3_gpu_tensor *destination,
                          size_t destination_offset,
                          const h3_gpu_tensor *source, size_t source_offset,
                          size_t elements);
int h3_gpu_copy_fp16_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                          size_t destination_offset,
                          const h3_gpu_tensor *source, size_t source_offset,
                          size_t elements);
/* Pack one VDN attention window in a single conversion dispatch. The four
 * segment arrays describe the text prefix, temporal window, and edge rows. */
int h3_gpu_pack_vdn_window_fp16(
                          h3_gpu *gpu,
                          h3_gpu_tensor *query_destination,
                          h3_gpu_tensor *key_destination,
                          h3_gpu_tensor *value_destination,
                          const h3_gpu_tensor *query_source,
                          const h3_gpu_tensor *key_source,
                          const h3_gpu_tensor *value_source,
                          uint32_t query_source_row,
                          uint32_t query_destination_row,
                          uint32_t query_rows,
                          const uint32_t *kv_source_rows,
                          const uint32_t *kv_destination_rows,
                          const uint32_t *kv_segment_rows,
                          uint32_t kv_segments,
                          uint32_t destination_batch,
                          uint32_t query_sequence, uint32_t kv_sequence,
                          uint32_t heads,
                          uint32_t head_dim,
                          int source_head_major, uint32_t source_sequence);
/* Convert row-major BF16 [row, head, dim] slices into batched, head-major
 * FP16 storage [batch, head, sequence, dim] for direct attention kernels. */
int h3_gpu_pack_bf16_fp16_head_major(
                          h3_gpu *gpu, h3_gpu_tensor *destination,
                          const h3_gpu_tensor *source, uint32_t source_row,
                          uint32_t destination_batch,
                          uint32_t destination_row, uint32_t sequence,
                          uint32_t rows, uint32_t heads, uint32_t head_dim);
int h3_gpu_pack_bf16_fp16_head_major_pair(
                          h3_gpu *gpu,
                          h3_gpu_tensor *first_destination,
                          h3_gpu_tensor *second_destination,
                          const h3_gpu_tensor *first_source,
                          const h3_gpu_tensor *second_source,
                          uint32_t source_row, uint32_t destination_batch,
                          uint32_t destination_row, uint32_t sequence,
                          uint32_t rows, uint32_t heads, uint32_t head_dim);
/* Copy BF16 rows into head-major BF16 storage. SOURCE_HEAD_MAJOR selects a
 * head-major source layout; SOURCE_SEQUENCE is its full row stride. */
int h3_gpu_pack_bf16_head_major_source(
                          h3_gpu *gpu, h3_gpu_tensor *destination,
                          const h3_gpu_tensor *source, uint32_t source_row,
                          uint32_t destination_batch,
                          uint32_t destination_row, uint32_t sequence,
                          uint32_t rows, uint32_t heads, uint32_t head_dim,
                          int source_head_major, uint32_t source_sequence);
int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3_gpu_tensor *source, size_t source_offset,
                    size_t elements);
/* In-place BF16 W += scale * B @ A over a contiguous output-row slice.
 * A is [rank, columns], B is [output_rows, rank]. */
int h3_gpu_lora_merge_bf16(h3_gpu *gpu, h3_gpu_tensor *weight,
                           const h3_gpu_tensor *a,
                           const h3_gpu_tensor *b,
                           uint32_t base_rows, uint32_t columns,
                           uint32_t output_offset,
                           uint32_t output_rows, uint32_t rank,
                           float scale);
/* Merge one Diffusers Q/K/V adapter into h3.c's checkpoint-native
 * [head, q/k/v, dimension] row layout. STREAM is 0, 1, or 2. */
int h3_gpu_lora_merge_grouped_qkv_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *weight,
                           const h3_gpu_tensor *a,
                           const h3_gpu_tensor *b,
                           uint32_t columns, uint32_t heads,
                           uint32_t head_dim, uint32_t stream,
                           uint32_t rank, float scale);
int h3_gpu_lora_merge_swap_halves_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *weight,
                           const h3_gpu_tensor *a,
                           const h3_gpu_tensor *b,
                           uint32_t columns, uint32_t half_rows,
                           uint32_t rank, float scale);
int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon);
int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon);
int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot);
int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon);
int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale);
int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width);
int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width);
int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon);
int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon);

/* H3 AudioVAE uses time-major [batch,length,channels] activations and stores
 * Conv1d/ConvTranspose1d weights in PyTorch OIK/IOK order respectively. */
int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation);
int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation);
int h3_gpu_conv_transpose1d_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding);
int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner);
int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements);
int h3_gpu_alias_free_snake_f32(
                          h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *alpha_log,
                          const h3_gpu_tensor *beta_log,
                          const h3_gpu_tensor *upsample_filter,
                          const h3_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels);
int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels);
int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim);
int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query,
                       const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence, uint32_t heads,
                       uint32_t head_dim, float scale);
int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim);
int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements);
int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum);

/* Visual-VAE encoder tensors use channels-last [B,T,H,W,C] storage. Spatial
 * padding reflects pixels while temporal front padding is zero-filled. */
int h3_gpu_vae_encoder_pad_f32(
                    h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after);
int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width);
int h3_gpu_vae_encoder_group_norm_silu_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon);

/* Portable BF16 storage path. Arithmetic accumulates in F32 and rounds at
 * operation boundaries, matching the released checkpoint's compute dtype. */
int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim);
/* Prefer native BF16 TensorOps in bounded row chunks when that backend is
 * available, otherwise retain the ordinary MPSGraph linear fallback. */
int h3_gpu_linear_bf16_split_rows(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim,
                       uint32_t split_rows);
int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim);
/* Experimental M5 Metal 4 paired FC1/SwiGLU plus direct FC2 path. Available
 * only when the context was created with H3_NAX=mlp. */
int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim);
/* Experimental M5 Metal 4 int8 MLP. Weights use one F32 scale per output
 * channel; activations are quantized dynamically with one F32 scale per row. */
int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns);
int h3_gpu_quantize_weight_int8_padded(h3_gpu *gpu,
                                h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t padded_rows, uint32_t columns);
int h3_gpu_linear_int8_56_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            const h3_gpu_tensor *bias,
                            uint32_t rows, uint32_t input_dim,
                            int input_is_quantized);
/* VDN beta projection over a contiguous row slice in a larger activation.
 * With input_is_quantized set, input_row selects matching int8/scales rows. */
int h3_gpu_linear_int8_56_bf16_offset(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            const h3_gpu_tensor *bias,
                            uint32_t input_row, uint32_t rows,
                            uint32_t input_dim, int input_is_quantized);
int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales);
int h3_gpu_linear_int8_prequantized_bf16(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales);
int h3_gpu_linear_int8_prequantized_bf16_offset(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t input_row, uint32_t rows,
                            uint32_t input_dim, uint32_t output_dim,
                            int use_slower_uncached_int8_scales);
/* Fixed-shape VDN output projection that adds its result in-place to an
 * existing BF16 residual row range, avoiding a separate staging tensor and
 * residual-add pass. */
int h3_gpu_linear_int8_prequantized_bf16_add_offset(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t output_row, uint32_t rows,
                            uint32_t input_dim, uint32_t output_dim);
int h3_gpu_linear_int8_bias_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            const h3_gpu_tensor *bias,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales);
/* Consume SDPA's native [head,row,dimension] BF16 layout without a full
 * BF16 transpose, gathering directly into the projection's row-major int8. */
int h3_gpu_linear_int8_head_major_bf16(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim);
int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16,
                         const h3_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k,
                         int use_int8_row_fc2,
                         int input_is_quantized);
int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon);
int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon);
int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate);
int h3_gpu_vision_qkv_rope_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half);
int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon);
int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon);
int h3_gpu_adaln_linear_bf16(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      h3_gpu_tensor *inverse,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon);
int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot);
int h3_gpu_gate_adaln_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon);
int h3_gpu_gate_adaln_quantize_int8(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *quantized_output,
                     h3_gpu_tensor *quantized_scales,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t padded_rows, uint32_t width, uint32_t slots,
                     uint32_t gate_slot, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon);
int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon);
/* H3 checkpoint QKV rows are [head, q/k/v, dimension], unlike the
 * conventional [q/k/v, head, dimension] layout accepted above. */
int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon);
/* Project grouped H3 QKV and apply its exact Q/K norm/RoPE boundary. Metal 4
 * may route projections directly into the attention layout; other devices
 * retain the ordinary two calls. */
int h3_gpu_grouped_qkv_linear_rope_bf16(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon);
int h3_gpu_grouped_qkv_linear_rope_int8(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *quantized_input,
                                 h3_gpu_tensor *input_scales,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *weight_scales,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales);
/* VDN variant: retain the unnormalized grouped QKV stream for its
 * convolutional feature path while writing normalized row-major Q/K/V. */
int h3_gpu_grouped_qkv_linear_rope_int8_vdn(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *raw_qkv,
                                 h3_gpu_tensor *quantized_input,
                                 h3_gpu_tensor *input_scales,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *weight_scales,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales);
int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
/* Dense cross-attention used by the VDN window decomposition. Inputs and
 * output are row-major [rows, heads, head_dim]. */
int h3_gpu_cross_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t query_sequence,
                     uint32_t kv_sequence, uint32_t heads,
                     uint32_t head_dim, float scale);
/* As above, with each tensor storing `batch` contiguous sequences. */
int h3_gpu_cross_sdpa_batched_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t batch,
                     uint32_t query_sequence, uint32_t kv_sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
/* Inputs/output use the same 16-bit arenas but contain IEEE FP16 bit patterns. */
int h3_gpu_cross_sdpa_batched_fp16_storage(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t batch,
                     uint32_t query_sequence, uint32_t kv_sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
/* Head-major IEEE FP16 inputs; output remains head-major IEEE FP16 storage. */
int h3_gpu_cross_sdpa_batched_fp16_head_major_storage(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t batch,
                     uint32_t query_sequence, uint32_t kv_sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
/* Dense BF16 cross-attention with row-major inputs and head-major output. */
int h3_gpu_cross_sdpa_bf16_head_major_output(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t query_sequence,
                     uint32_t kv_sequence, uint32_t heads,
                     uint32_t head_dim, float scale);
int h3_gpu_vdn_gate_heads_bf16(h3_gpu *gpu, h3_gpu_tensor *heads,
                     const h3_gpu_tensor *logits, uint32_t rows,
                     uint32_t head_count, uint32_t head_dim);
int h3_gpu_vdn_copy_gate_heads_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *destination,
                     uint32_t destination_row,
                     const h3_gpu_tensor *source, uint32_t source_row,
                     const h3_gpu_tensor *logits, uint32_t logit_row,
                     uint32_t rows, uint32_t head_count, uint32_t head_dim,
                     int source_is_fp16);
/* Variant with an explicit source sequence stride for head-major FP16
 * sources. This is used when padded attention batches share one graph shape. */
int h3_gpu_vdn_copy_gate_heads_bf16_strided(
                     h3_gpu *gpu, h3_gpu_tensor *destination,
                     uint32_t destination_row,
                     const h3_gpu_tensor *source, uint32_t source_row,
                     const h3_gpu_tensor *logits, uint32_t logit_row,
                     uint32_t rows, uint32_t head_count, uint32_t head_dim,
                     int source_is_fp16, uint32_t source_sequence);
/* Gate a head-major SDPA result and quantize directly into the row-major int8
 * arena used by the attention-output projection. SOURCE_BATCH selects a
 * packed window; SOURCE_SEQUENCE is the per-batch source stride. */
int h3_gpu_vdn_gate_quantize_int8_head_major(
                     h3_gpu *gpu, h3_gpu_tensor *quantized_output,
                     h3_gpu_tensor *quantized_scales,
                     const h3_gpu_tensor *source,
                     const h3_gpu_tensor *logits,
                     uint32_t destination_row, uint32_t source_batch,
                     uint32_t source_row, uint32_t logit_row,
                     uint32_t rows, uint32_t head_count, uint32_t head_dim,
                     int source_is_fp16, uint32_t source_sequence);
int h3_gpu_vdn_text_features_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *key,
                     h3_gpu_tensor *value, const h3_gpu_tensor *grouped_qkv,
                     uint32_t source_row, uint32_t rows,
                     uint32_t heads, uint32_t head_dim);
int h3_gpu_vdn_query_feature_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *query_head_major,
                     const h3_gpu_tensor *grouped_qkv,
                     uint32_t source_row, uint32_t frames,
                     uint32_t tokens_per_frame, uint32_t heads,
                     uint32_t head_dim);
int h3_gpu_vdn_spatial_feature_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *spatial,
                     const h3_gpu_tensor *grouped_qkv,
                     const h3_gpu_tensor *weight, uint32_t source_row,
                     uint32_t frames, uint32_t height, uint32_t width,
                     uint32_t heads, uint32_t head_dim, uint32_t stream);
/* Fused K/V spatial convolution. Both streams read the grouped QKV tile in
 * one dispatch, sharing the expensive source loads. The pair path is
 * opt-in because its vector accumulation order is intentionally explicit. */
int h3_gpu_vdn_spatial_feature_bf16_pair(
                     h3_gpu *gpu, h3_gpu_tensor *key_spatial,
                     h3_gpu_tensor *value_spatial,
                     const h3_gpu_tensor *grouped_qkv,
                     const h3_gpu_tensor *key_weight,
                     const h3_gpu_tensor *value_weight,
                     uint32_t source_row, uint32_t frames,
                     uint32_t height, uint32_t width, uint32_t heads,
                     uint32_t head_dim);
int h3_gpu_vdn_temporal_feature_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *feature,
                     const h3_gpu_tensor *spatial,
                     const h3_gpu_tensor *weight, uint32_t frames,
                     uint32_t tokens_per_frame, uint32_t heads,
                     uint32_t head_dim, int l2_normalize);
/* Fused K/V temporal convolution. KEY_FEATURE receives the L2-normalized
 * result; VALUE_FEATURE receives the unnormalized SiLU result. */
int h3_gpu_vdn_temporal_feature_bf16_pair(
                     h3_gpu *gpu, h3_gpu_tensor *key_feature,
                     h3_gpu_tensor *value_feature,
                     const h3_gpu_tensor *key_spatial,
                     const h3_gpu_tensor *value_spatial,
                     const h3_gpu_tensor *key_weight,
                     const h3_gpu_tensor *value_weight, uint32_t frames,
                     uint32_t tokens_per_frame, uint32_t heads,
                     uint32_t head_dim);
/* Fused K/V temporal convolution plus the VDN query feature.  The query is
 * read from the raw grouped QKV stream while the K/V pair is produced, so the
 * standalone query-feature pass can be omitted.  SOURCE_ROW is the first row
 * in GROUPED_QKV corresponding to the local temporal slice. */
int h3_gpu_vdn_temporal_feature_bf16_pair_query(
                     h3_gpu *gpu, h3_gpu_tensor *key_feature,
                     h3_gpu_tensor *value_feature, h3_gpu_tensor *query_feature,
                     const h3_gpu_tensor *grouped_qkv,
                     const h3_gpu_tensor *key_spatial,
                     const h3_gpu_tensor *value_spatial,
                     const h3_gpu_tensor *key_weight,
                     const h3_gpu_tensor *value_weight, uint32_t source_row,
                     uint32_t frames, uint32_t tokens_per_frame,
                     uint32_t heads, uint32_t head_dim);
/* Fused FP16 statistics preparation for the fixed H3 VDN shape.  Temporal
 * K/V and raw-Q features are computed once, then written directly into the
 * batched head-major FP16 layouts consumed by the MPS products.  The packed
 * destinations retain 16-bit storage but may be backed by BF16/F32 tensors
 * whose public dtype is used elsewhere in the activation arena. */
int h3_gpu_vdn_temporal_feature_pair_query_stats_fp16(
                     h3_gpu *gpu, h3_gpu_tensor *packed_key,
                     h3_gpu_tensor *packed_scaled_key,
                     h3_gpu_tensor *packed_scaled_value,
                     h3_gpu_tensor *query_feature,
                     const h3_gpu_tensor *grouped_qkv,
                     const h3_gpu_tensor *key_spatial,
                     const h3_gpu_tensor *value_spatial,
                     const h3_gpu_tensor *key_weight,
                     const h3_gpu_tensor *value_weight,
                     const h3_gpu_tensor *beta_logits, uint32_t source_row,
                     uint32_t frames, uint32_t tokens_per_frame,
                     uint32_t heads, uint32_t head_dim);
int h3_gpu_vdn_frame_mean_f32(
                     h3_gpu *gpu, h3_gpu_tensor *mean,
                     const h3_gpu_tensor *input, uint32_t frames,
                     uint32_t tokens_per_frame, uint32_t width);
/* Same reduction over a contiguous row slice in a larger BF16 activation. */
int h3_gpu_vdn_frame_mean_f32_offset(
                     h3_gpu *gpu, h3_gpu_tensor *mean,
                     const h3_gpu_tensor *input, size_t input_offset,
                     uint32_t frames, uint32_t tokens_per_frame,
                     uint32_t width);
/* Compute the VDN per-frame mean from an existing row-quantized activation.
 * INPUT_ROW is measured in rows of WIDTH elements; the result is dequantized
 * with the corresponding per-row scales. */
int h3_gpu_vdn_frame_mean_int8_f32_offset(
                     h3_gpu *gpu, h3_gpu_tensor *mean,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *input_scales,
                     uint32_t input_row, uint32_t frames,
                     uint32_t tokens_per_frame, uint32_t width);
int h3_gpu_vdn_linear_f32_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input_f32,
                     const h3_gpu_tensor *weight_bf16,
                     uint32_t rows, uint32_t input_dim,
                     uint32_t output_dim);
/* Fused alpha-up projection and FrameKDA activation. This writes the final
 * alpha values directly, avoiding the intermediate delta read/write pass. */
int h3_gpu_vdn_linear_alpha_f32_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input_f32,
                     const h3_gpu_tensor *weight_bf16,
                     const h3_gpu_tensor *dt_bias_bf16,
                     const h3_gpu_tensor *a_log_bf16,
                     uint32_t rows, uint32_t input_dim,
                     uint32_t output_dim, uint32_t heads,
                     uint32_t head_dim);
int h3_gpu_vdn_alpha_f32(
                     h3_gpu *gpu, h3_gpu_tensor *alpha,
                     const h3_gpu_tensor *delta,
                     const h3_gpu_tensor *dt_bias,
                     const h3_gpu_tensor *a_log,
                     uint32_t frames, uint32_t heads, uint32_t head_dim);
int h3_gpu_vdn_statistics_f32(
                     h3_gpu *gpu, h3_gpu_tensor *a, h3_gpu_tensor *b,
                     const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value,
                     const h3_gpu_tensor *beta_logits,
                     uint32_t frames, uint32_t tokens_per_frame,
                     uint32_t heads, uint32_t head_dim);
int h3_gpu_vdn_statistics_fp16(
                     h3_gpu *gpu, h3_gpu_tensor *a, h3_gpu_tensor *b,
                     const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value,
                     const h3_gpu_tensor *beta_logits,
                     h3_gpu_tensor *packed_key,
                     h3_gpu_tensor *packed_scaled,
                     h3_gpu_tensor *product,
                     uint32_t frames, uint32_t tokens_per_frame,
                     uint32_t heads, uint32_t head_dim);
/* Consume already-packed FP16 VDN statistics inputs.  This skips the two
 * BF16->FP16 transpose/gate passes used by h3_gpu_vdn_statistics_fp16; the
 * packed tensors are validated by byte capacity because their storage can be
 * borrowed from BF16/F32 scratch buffers. */
int h3_gpu_vdn_statistics_fp16_packed(
                     h3_gpu *gpu, h3_gpu_tensor *a, h3_gpu_tensor *b,
                     const h3_gpu_tensor *packed_key,
                     const h3_gpu_tensor *packed_scaled_key,
                     const h3_gpu_tensor *packed_scaled_value,
                     h3_gpu_tensor *product, uint32_t frames,
                     uint32_t tokens_per_frame, uint32_t heads,
                     uint32_t head_dim);
int h3_gpu_vdn_solve_f32(
                     h3_gpu *gpu, h3_gpu_tensor *a_factor,
                     h3_gpu_tensor *injection,
                     h3_gpu_tensor *rhs, h3_gpu_tensor *solution,
                     const h3_gpu_tensor *alpha,
                     uint32_t frames, uint32_t heads, uint32_t head_dim);
/* Solve variant that writes the compact FP16 transition/injection banks into
 * scan_workspace as part of the final solve-pack pass.  The workspace is an
 * F32 tensor only for ownership/capacity accounting; its bytes are viewed as
 * half values by h3_gpu_vdn_scan_fp16_prepacked(). */
int h3_gpu_vdn_solve_f32_fp16_scan(
                     h3_gpu *gpu, h3_gpu_tensor *a_factor,
                     h3_gpu_tensor *injection,
                     h3_gpu_tensor *rhs, h3_gpu_tensor *solution,
                     const h3_gpu_tensor *alpha,
                     h3_gpu_tensor *scan_workspace,
                     uint32_t frames, uint32_t heads, uint32_t head_dim);
/* Variant that also seeds PREFIX/SUFFIX's compact FP16 banks in the solve
 * pack, allowing h3_gpu_vdn_scan_fp16_prepacked_initialized() to skip its
 * two initialization blits.  The FP32 transition/injection outputs are not
 * needed by that prepacked scan and may be left unspecified. */
int h3_gpu_vdn_solve_f32_fp16_scan_to_buffers(
                     h3_gpu *gpu, h3_gpu_tensor *a_factor,
                     h3_gpu_tensor *injection,
                     h3_gpu_tensor *rhs, h3_gpu_tensor *solution,
                     const h3_gpu_tensor *alpha,
                     h3_gpu_tensor *scan_workspace,
                     h3_gpu_tensor *scan_prefix,
                     h3_gpu_tensor *scan_suffix,
                     uint32_t frames, uint32_t heads, uint32_t head_dim);
int h3_gpu_vdn_scale_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     float scale);
int h3_gpu_vdn_decay_f32(
                     h3_gpu *gpu, h3_gpu_tensor *before_decay,
                     h3_gpu_tensor *after_decay,
                     const h3_gpu_tensor *alpha,
                     uint32_t frames, uint32_t heads, uint32_t head_dim);
int h3_gpu_vdn_scan_f32(
                     h3_gpu *gpu, h3_gpu_tensor *prefix,
                     h3_gpu_tensor *suffix,
                     const h3_gpu_tensor *injection,
                     const h3_gpu_tensor *solution,
                     const h3_gpu_tensor *text_state,
                     uint32_t frames, uint32_t heads, uint32_t head_dim);
/* Optional compact scan.  The transition/injection sources remain F32, while
 * PREFIX/SUFFIX and the temporary scan workspace are interpreted as FP16
 * storage for the MPS matrix products.  The workspace is only needed until
 * this call completes and can alias the VDN solve RHS arena. */
int h3_gpu_vdn_scan_fp16(
                     h3_gpu *gpu, h3_gpu_tensor *prefix,
                     h3_gpu_tensor *suffix,
                     const h3_gpu_tensor *injection,
                     const h3_gpu_tensor *solution,
                     h3_gpu_tensor *scratch,
                     const h3_gpu_tensor *text_state,
                     uint32_t frames, uint32_t heads, uint32_t head_dim);
/* Same scan, but scan_workspace already contains the compact FP16 transition
 * and injection banks produced by h3_gpu_vdn_solve_f32_fp16_scan(). */
int h3_gpu_vdn_scan_fp16_prepacked(
                     h3_gpu *gpu, h3_gpu_tensor *prefix,
                     h3_gpu_tensor *suffix,
                     const h3_gpu_tensor *injection,
                     const h3_gpu_tensor *solution,
                     h3_gpu_tensor *scan_workspace,
                     const h3_gpu_tensor *text_state,
                     uint32_t frames, uint32_t heads, uint32_t head_dim);
/* Same as the prepacked scan, but PREFIX/SUFFIX already contain their FP16
 * injection banks from h3_gpu_vdn_solve_f32_fp16_scan_to_buffers(). */
int h3_gpu_vdn_scan_fp16_prepacked_initialized(
                     h3_gpu *gpu, h3_gpu_tensor *prefix,
                     h3_gpu_tensor *suffix,
                     const h3_gpu_tensor *injection,
                     const h3_gpu_tensor *solution,
                     h3_gpu_tensor *scan_workspace,
                     const h3_gpu_tensor *text_state,
                     uint32_t frames, uint32_t heads, uint32_t head_dim);
int h3_gpu_vdn_gather_state_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *state,
                     const h3_gpu_tensor *prefix,
                     const h3_gpu_tensor *suffix,
                     const h3_gpu_tensor *alpha,
                     const h3_gpu_tensor *text_state,
                     uint32_t frames, uint32_t heads, uint32_t head_dim);
int h3_gpu_vdn_gather_state_fp16(
                     h3_gpu *gpu, h3_gpu_tensor *state,
                     const h3_gpu_tensor *prefix,
                     const h3_gpu_tensor *suffix,
                     const h3_gpu_tensor *alpha,
                     const h3_gpu_tensor *text_state,
                     uint32_t frames, uint32_t heads, uint32_t head_dim);
int h3_gpu_vdn_gather_state_bf16_decay(
                     h3_gpu *gpu, h3_gpu_tensor *state,
                     const h3_gpu_tensor *prefix,
                     const h3_gpu_tensor *suffix,
                     const h3_gpu_tensor *alpha,
                     const h3_gpu_tensor *text_state,
                     const h3_gpu_tensor *before_decay,
                     const h3_gpu_tensor *after_decay,
                     uint32_t frames, uint32_t heads, uint32_t head_dim);
int h3_gpu_vdn_gather_state_fp16_decay(
                     h3_gpu *gpu, h3_gpu_tensor *state,
                     const h3_gpu_tensor *prefix,
                     const h3_gpu_tensor *suffix,
                     const h3_gpu_tensor *alpha,
                     const h3_gpu_tensor *text_state,
                     const h3_gpu_tensor *before_decay,
                     const h3_gpu_tensor *after_decay,
                     uint32_t frames, uint32_t heads, uint32_t head_dim);
int h3_gpu_vdn_readout_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *output_head_major,
                     const h3_gpu_tensor *query_head_major,
                     const h3_gpu_tensor *state,
                     uint32_t frames, uint32_t tokens_per_frame,
                     uint32_t heads, uint32_t head_dim);
int h3_gpu_vdn_epilogue_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *output_token_major,
                     const h3_gpu_tensor *readout_head_major,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_logits,
                     uint32_t frames, uint32_t tokens_per_frame,
                     uint32_t heads, uint32_t head_dim, float epsilon);
/* Fused VDN epilogue and dynamic row quantization.  The opt-in path writes
 * the row-major int8 activation/scales consumed by the VDN output projection
 * directly, avoiding a full-width BF16 staging buffer and reread. */
int h3_gpu_vdn_epilogue_quantize_int8(
                     h3_gpu *gpu, h3_gpu_tensor *output_int8,
                     h3_gpu_tensor *output_scales,
                     const h3_gpu_tensor *readout_head_major,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_logits,
                     uint32_t frames, uint32_t tokens_per_frame,
                     uint32_t heads, uint32_t head_dim, float epsilon);
int h3_gpu_vdn_add_projected_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *destination,
                     uint32_t destination_row,
                     const h3_gpu_tensor *source, uint32_t rows,
                     uint32_t width);
/* Preserve SDPA's native [head,row,dimension] output for an immediately
 * following layout-aware projection. */
int h3_gpu_sdpa_bf16_head_major_output(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width);
int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width);
int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu,
                             h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon);
int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon);
int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim);
int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim,
                           float scale);
int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements);
int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements);
int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           size_t input_offset,
                           h3_gpu_tensor *original,
                           size_t original_offset,
                           h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width);
int h3_gpu_token_pool_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t input_rows, uint32_t rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon);
int h3_gpu_token_expand_delta_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width,
                           uint32_t exact_prefix_rows,
                           float update_scale);
int h3_gpu_token_expand_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t rows, uint32_t reduced_rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t exact_prefix_rows, float update_scale,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon);
/* Apply one Euler step to an F32 sample range from BF16 velocity caches:
 * sample += delta * (last + ratio * (last - previous)). */
int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio);
int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements);

#endif
