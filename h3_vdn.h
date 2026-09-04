#ifndef H3_VDN_H
#define H3_VDN_H

#include "h3_gpu.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_vdn h3_vdn;

typedef struct {
    h3_gpu_tensor *alpha_a_log;
    h3_gpu_tensor *alpha_down;
    h3_gpu_tensor *alpha_dt_bias;
    h3_gpu_tensor *alpha_up;
    h3_gpu_tensor *beta;
    h3_gpu_tensor *beta_int8;
    h3_gpu_tensor *beta_scales;
    h3_gpu_tensor *norm;
    h3_gpu_tensor *output_gate_down;
    h3_gpu_tensor *output_gate_down_int8;
    h3_gpu_tensor *output_gate_down_scales;
    h3_gpu_tensor *output_gate_up_bias;
    h3_gpu_tensor *output_gate_up;
    h3_gpu_tensor *output_gate_up_int8;
    h3_gpu_tensor *output_gate_up_scales;
    h3_gpu_tensor *k_spatial;
    h3_gpu_tensor *k_temporal;
    h3_gpu_tensor *v_spatial;
    h3_gpu_tensor *v_temporal;
    h3_gpu_tensor *softmax_gate_bias;
    h3_gpu_tensor *softmax_gate;
    h3_gpu_tensor *softmax_gate_int8;
    h3_gpu_tensor *softmax_gate_scales;
    h3_gpu_tensor *output;
    h3_gpu_tensor *output_int8;
    h3_gpu_tensor *output_scales;
} h3_vdn_block;

/* Open one released OpenVDN exploded checkpoint (Stage B or Stage DMD).
 * This validates the checkpoint-bound hybrid architecture and opens the
 * safetensor headers without mapping the tensor payloads. */
h3_vdn *h3_vdn_open(const char *directory, char *error, size_t error_size);
void h3_vdn_free(h3_vdn *vdn);
int h3_vdn_recommended_steps(const char *directory,
                             char *error, size_t error_size);

int h3_vdn_is_turbo(const h3_vdn *vdn);
const h3_weight_store *h3_vdn_branch_weights(const h3_vdn *vdn);
int h3_vdn_load_block(const h3_vdn *vdn, h3_gpu *gpu, unsigned block,
                      h3_vdn_block *weights,
                      char *error, size_t error_size);
void h3_vdn_free_block(h3_vdn_block *weights);

/* Copy BASE into a writable BF16 tensor and merge every adapter that contains
 * TARGET. TARGET names use the Diffusers checkpoint spelling. OUTPUT_OFFSET is
 * measured in rows, allowing the separate q/k/v adapters to update slices of
 * h3.c's combined QKV matrix. */
h3_gpu_tensor *h3_vdn_adapt_weight_bf16(
    const h3_vdn *vdn, h3_gpu *gpu, const h3_gpu_tensor *base,
    uint32_t base_rows, uint32_t columns, const char *target,
    uint32_t output_offset, uint32_t output_rows,
    char *error, size_t error_size);

/* Merge another target into an already writable matrix returned above. */
int h3_vdn_merge_target_bf16(
    const h3_vdn *vdn, h3_gpu *gpu, h3_gpu_tensor *weight,
    uint32_t base_rows, uint32_t columns, const char *target,
    uint32_t output_offset, uint32_t output_rows,
    char *error, size_t error_size);

h3_gpu_tensor *h3_vdn_adapt_grouped_qkv_bf16(
    const h3_vdn *vdn, h3_gpu *gpu, const h3_gpu_tensor *base,
    uint32_t columns, uint32_t heads, uint32_t head_dim,
    const char *q_target, const char *k_target, const char *v_target,
    char *error, size_t error_size);
h3_gpu_tensor *h3_vdn_adapt_swiglu_bf16(
    const h3_vdn *vdn, h3_gpu *gpu, const h3_gpu_tensor *base,
    uint32_t columns, uint32_t half_rows, const char *target,
    char *error, size_t error_size);

#endif
