#include "h3_vdn.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

struct h3_vdn {
    char *directory;
    h3_weight_store *branch;
    h3_weight_store *adapter_default;
    h3_weight_store *adapter_turbo;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static char *path_join(const char *root, const char *suffix) {
    size_t size = strlen(root) + strlen(suffix) + 2;
    char *path = malloc(size);
    if (path) snprintf(path, size, "%s/%s", root, suffix);
    return path;
}

static char *read_text(const char *path, char *error, size_t error_size) {
    FILE *stream = fopen(path, "rb");
    if (!stream) {
        fail(error, error_size, "%s: %s", path, strerror(errno));
        return NULL;
    }
    if (fseek(stream, 0, SEEK_END) != 0) {
        fail(error, error_size, "%s: cannot seek", path);
        fclose(stream);
        return NULL;
    }
    long length = ftell(stream);
    if (length < 0 || length > 1024 * 1024 || fseek(stream, 0, SEEK_SET) != 0) {
        fail(error, error_size, "%s: invalid metadata size", path);
        fclose(stream);
        return NULL;
    }
    char *text = malloc((size_t)length + 1);
    if (!text) {
        fail(error, error_size, "out of memory reading %s", path);
        fclose(stream);
        return NULL;
    }
    size_t read = fread(text, 1, (size_t)length, stream);
    fclose(stream);
    if (read != (size_t)length) {
        fail(error, error_size, "%s: incomplete metadata read", path);
        free(text);
        return NULL;
    }
    text[length] = '\0';
    return text;
}

static int validate_spec(const char *directory, char *error,
                         size_t error_size);

static int regular_file(const char *path) {
    struct stat status;
    return stat(path, &status) == 0 && S_ISREG(status.st_mode);
}

int h3_vdn_recommended_steps(const char *directory,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!directory || !*directory ||
        !validate_spec(directory, error, error_size)) return -1;
    char *turbo = path_join(directory,
                            "adapters/turbo/adapter_model.safetensors");
    if (!turbo) {
        fail(error, error_size, "out of memory resolving VDN adapter");
        return -1;
    }
    int is_turbo = regular_file(turbo);
    free(turbo);
    if (!is_turbo) return 50;
    char *metadata = path_join(directory, "metadata.json");
    if (!metadata) {
        fail(error, error_size, "out of memory resolving VDN metadata");
        return -1;
    }
    char *text = read_text(metadata, error, error_size);
    free(metadata);
    if (!text) return -1;
    const char *field = strstr(text, "\"turbo_num_steps\"");
    const char *colon = field ? strchr(field, ':') : NULL;
    char *tail = NULL;
    long steps = colon ? strtol(colon + 1, &tail, 10) : -1;
    free(text);
    if (!colon || tail == colon + 1 || steps < 2 || steps > 1000) {
        fail(error, error_size,
             "turbo VDN metadata has no valid turbo_num_steps");
        return -1;
    }
    return (int)steps;
}

static int validate_spec(const char *directory, char *error,
                         size_t error_size) {
    char *path = path_join(directory, "model_spec.json");
    if (!path) {
        fail(error, error_size, "out of memory resolving VDN model spec");
        return 0;
    }
    char *text = read_text(path, error, error_size);
    free(path);
    if (!text) return 0;
    static const char *required[] = {
        "\"type\": \"hybrid_attention\"",
        "\"version\": 2",
        "\"anchor_frames\": \"both\"",
        "\"enable_softmax_gate\": true",
        "\"delta_rule\": \"vdn_solve\"",
        "\"enable_text_state\": true",
        "\"a_fp32\": true",
        "\"linear_head_dim\": 128",
        "\"bridge\": \"alpha\"",
        "\"chunk\": 5",
        "\"radius\": 1"
    };
    int ok = 1;
    for (size_t index = 0; index < sizeof(required) / sizeof(*required);
         index++) {
        if (!strstr(text, required[index])) {
            fail(error, error_size,
                 "unsupported VDN model spec (missing %s)", required[index]);
            ok = 0;
            break;
        }
    }
    char *short_conv = strstr(text, "\"short_conv\"");
    char *short_conv_end = short_conv ? strchr(short_conv, '}') : NULL;
    char *short_k = short_conv ? strstr(short_conv, "\"k\"") : NULL;
    char *short_v = short_conv ? strstr(short_conv, "\"v\"") : NULL;
    char *short_q = short_conv ? strstr(short_conv, "\"q\"") : NULL;
    if (ok && (!short_conv_end || !short_k || short_k >= short_conv_end ||
               !short_v || short_v >= short_conv_end ||
               (short_q && short_q < short_conv_end))) {
        fail(error, error_size,
             "unsupported VDN model spec (short_conv must target k and v only)");
        ok = 0;
    }
    free(text);
    return ok;
}

h3_vdn *h3_vdn_open(const char *directory, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!directory || !*directory ||
        !validate_spec(directory, error, error_size)) return NULL;
    h3_vdn *vdn = calloc(1, sizeof(*vdn));
    if (!vdn) {
        fail(error, error_size, "out of memory opening VDN checkpoint");
        return NULL;
    }
    vdn->directory = strdup(directory);
    char *branch = path_join(directory, "linear_branch");
    char *normal = path_join(directory, "adapters/default");
    char *turbo = path_join(directory, "adapters/turbo");
    if (!vdn->directory || !branch || !normal || !turbo) {
        fail(error, error_size, "out of memory resolving VDN checkpoint paths");
        free(branch); free(normal); free(turbo);
        h3_vdn_free(vdn);
        return NULL;
    }
    vdn->branch = h3_weight_store_open(branch, error, error_size);
    if (vdn->branch)
        vdn->adapter_default = h3_weight_store_open(
            normal, error, error_size);
    char *turbo_file = path_join(turbo, "adapter_model.safetensors");
    if (vdn->adapter_default && turbo_file && regular_file(turbo_file))
        vdn->adapter_turbo = h3_weight_store_open(turbo, error, error_size);
    free(turbo_file);
    free(branch); free(normal); free(turbo);
    if (!vdn->branch || !vdn->adapter_default ||
        (error && error_size && error[0])) {
        h3_vdn_free(vdn);
        return NULL;
    }
    return vdn;
}

void h3_vdn_free(h3_vdn *vdn) {
    if (!vdn) return;
    h3_weight_store_free(vdn->branch);
    h3_weight_store_free(vdn->adapter_default);
    h3_weight_store_free(vdn->adapter_turbo);
    free(vdn->directory);
    free(vdn);
}

int h3_vdn_is_turbo(const h3_vdn *vdn) {
    return vdn && vdn->adapter_turbo != NULL;
}

const h3_weight_store *h3_vdn_branch_weights(const h3_vdn *vdn) {
    return vdn ? vdn->branch : NULL;
}

static h3_gpu_tensor *branch_bf16(const h3_vdn *vdn, h3_gpu *gpu,
                                  unsigned block, const char *suffix,
                                  int ndim, const uint64_t *shape,
                                  char *error, size_t error_size) {
    char name[192];
    int length = snprintf(name, sizeof(name),
                          "transformer_blocks.%u.attn.%s", block, suffix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        fail(error, error_size, "VDN branch tensor name is too long");
        return NULL;
    }
    return h3_weight_load_bf16(vdn->branch, gpu, name, ndim, shape,
                               error, error_size);
}

int h3_vdn_load_block(const h3_vdn *vdn, h3_gpu *gpu, unsigned block,
                      h3_vdn_block *weights,
                      char *error, size_t error_size) {
    if (!vdn || !gpu || !weights || block >= 50) {
        fail(error, error_size, "invalid VDN block load request");
        return 0;
    }
    uint64_t h56[] = {56};
    uint64_t h128[] = {128};
    uint64_t h7168[] = {7168};
    uint64_t down[] = {128, 5376};
    uint64_t up[] = {7168, 128};
    uint64_t beta[] = {56, 5376};
    uint64_t spatial[] = {7168, 1, 5, 5};
    uint64_t temporal[] = {7168, 1, 5};
    uint64_t output[] = {5376, 7168};
#define LOAD(field, suffix, ndim, shape) do {                                  \
    weights->field = branch_bf16(vdn, gpu, block, suffix, ndim, shape,         \
                                  error, error_size);                           \
    if (!weights->field) goto failed;                                           \
} while (0)
    LOAD(alpha_a_log, "linear_attention.alpha.A_log", 1, h56);
    LOAD(alpha_down, "linear_attention.alpha.down.weight", 2, down);
    LOAD(alpha_dt_bias, "linear_attention.alpha.dt_bias", 1, h7168);
    LOAD(alpha_up, "linear_attention.alpha.up.weight", 2, up);
    LOAD(beta, "linear_attention.beta_proj.weight", 2, beta);
    LOAD(norm, "linear_attention.norm.weight", 1, h128);
    LOAD(output_gate_down, "linear_attention.output_gate.down.weight", 2,
         down);
    LOAD(output_gate_up_bias, "linear_attention.output_gate.up.bias", 1,
         h7168);
    LOAD(output_gate_up, "linear_attention.output_gate.up.weight", 2, up);
    LOAD(k_spatial, "linear_attention.short_conv.k_sp.weight", 4, spatial);
    LOAD(k_temporal, "linear_attention.short_conv.k_tm.weight", 3, temporal);
    LOAD(v_spatial, "linear_attention.short_conv.v_sp.weight", 4, spatial);
    LOAD(v_temporal, "linear_attention.short_conv.v_tm.weight", 3, temporal);
    LOAD(softmax_gate_bias, "softmax_gate.up.bias", 1, h56);
    LOAD(softmax_gate, "softmax_gate.up.weight", 2, beta);
    LOAD(output, "to_out_linear.weight", 2, output);
#undef LOAD
    if (h3_gpu_has_int8_mlp(gpu) && !getenv("H3_DISABLE_VDN_INT8")) {
        weights->output_gate_down_int8 = h3_gpu_tensor_new_i8(
            gpu, (size_t)128 * 5376);
        weights->output_gate_down_scales = h3_gpu_tensor_new_f32(gpu, 128);
        weights->output_gate_up_int8 = h3_gpu_tensor_new_i8(
            gpu, (size_t)7168 * 128);
        weights->output_gate_up_scales = h3_gpu_tensor_new_f32(gpu, 7168);
        weights->output_int8 = h3_gpu_tensor_new_i8(
            gpu, (size_t)5376 * 7168);
        weights->output_scales = h3_gpu_tensor_new_f32(gpu, 5376);
        weights->beta_int8 = h3_gpu_tensor_new_i8(gpu, (size_t)64 * 5376);
        weights->beta_scales = h3_gpu_tensor_new_f32(gpu, 64);
        weights->softmax_gate_int8 = h3_gpu_tensor_new_i8(
            gpu, (size_t)64 * 5376);
        weights->softmax_gate_scales = h3_gpu_tensor_new_f32(gpu, 64);
        int ok = weights->beta_int8 && weights->beta_scales &&
            weights->softmax_gate_int8 && weights->softmax_gate_scales &&
            weights->output_gate_down_int8 &&
            weights->output_gate_down_scales && weights->output_gate_up_int8 &&
            weights->output_gate_up_scales && weights->output_int8 &&
            weights->output_scales && h3_gpu_begin(gpu);
        if (ok) ok = h3_gpu_quantize_weight_int8_padded(
            gpu, weights->beta_int8, weights->beta_scales, weights->beta,
            56, 64, 5376);
        if (ok) ok = h3_gpu_quantize_weight_int8_padded(
            gpu, weights->softmax_gate_int8, weights->softmax_gate_scales,
            weights->softmax_gate, 56, 64, 5376);
        if (ok) ok = h3_gpu_quantize_weight_int8(
            gpu, weights->output_gate_down_int8,
            weights->output_gate_down_scales, weights->output_gate_down,
            128, 5376);
        if (ok) ok = h3_gpu_quantize_weight_int8(
            gpu, weights->output_gate_up_int8,
            weights->output_gate_up_scales, weights->output_gate_up,
            7168, 128);
        if (ok) ok = h3_gpu_quantize_weight_int8(
            gpu, weights->output_int8, weights->output_scales,
            weights->output, 5376, 7168);
        if (ok) ok = h3_gpu_submit(gpu);
        if (!ok) {
            if (!error || !error_size || !error[0])
                fail(error, error_size, "cannot quantize VDN projections: %s",
                     h3_gpu_error(gpu));
            goto failed;
        }
    }
    return 1;
failed:
#undef LOAD
    h3_vdn_free_block(weights);
    return 0;
}

void h3_vdn_free_block(h3_vdn_block *weights) {
    if (!weights) return;
#define FREE(field) do {                                                        \
    h3_gpu_tensor_free(weights->field);                                         \
    weights->field = NULL;                                                      \
} while (0)
    FREE(alpha_a_log);
    FREE(alpha_down);
    FREE(alpha_dt_bias);
    FREE(alpha_up);
    FREE(beta);
    FREE(beta_int8);
    FREE(beta_scales);
    FREE(norm);
    FREE(output_gate_down);
    FREE(output_gate_down_int8);
    FREE(output_gate_down_scales);
    FREE(output_gate_up_bias);
    FREE(output_gate_up);
    FREE(output_gate_up_int8);
    FREE(output_gate_up_scales);
    FREE(k_spatial);
    FREE(k_temporal);
    FREE(v_spatial);
    FREE(v_temporal);
    FREE(softmax_gate_bias);
    FREE(softmax_gate);
    FREE(softmax_gate_int8);
    FREE(softmax_gate_scales);
    FREE(output);
    FREE(output_int8);
    FREE(output_scales);
#undef FREE
}

static int merge_from(const h3_weight_store *store, const char *adapter,
                      h3_gpu *gpu, h3_gpu_tensor *weight,
                      uint32_t base_rows, uint32_t columns,
                      const char *target, uint32_t output_offset,
                      uint32_t output_rows, int grouped_stream, int *found,
                      char *error, size_t error_size) {
    if (!store) return 1;
    char a_name[256], b_name[256];
    int a_length = snprintf(a_name, sizeof(a_name), "%s.lora_A.%s.weight",
                            target, adapter);
    int b_length = snprintf(b_name, sizeof(b_name), "%s.lora_B.%s.weight",
                            target, adapter);
    if (a_length < 0 || b_length < 0 ||
        (size_t)a_length >= sizeof(a_name) ||
        (size_t)b_length >= sizeof(b_name)) {
        fail(error, error_size, "VDN adapter target name is too long");
        return 0;
    }
    const h3_st_header *a_header = NULL, *b_header = NULL;
    const h3_st_tensor *a_meta = h3_weight_find(store, a_name, &a_header);
    const h3_st_tensor *b_meta = h3_weight_find(store, b_name, &b_header);
    if (!a_meta && !b_meta) return 1;
    *found = 1;
    if (!a_meta || !b_meta || a_meta->dtype != H3_DTYPE_BF16 ||
        b_meta->dtype != H3_DTYPE_BF16 || a_meta->ndim != 2 ||
        b_meta->ndim != 2 || a_meta->shape[1] != columns ||
        b_meta->shape[0] != output_rows ||
        a_meta->shape[0] != b_meta->shape[1] ||
        a_meta->shape[0] == 0 || a_meta->shape[0] > UINT32_MAX) {
        fail(error, error_size, "VDN LoRA tensor schema mismatch for %s", target);
        return 0;
    }
    uint32_t rank = (uint32_t)a_meta->shape[0];
    uint64_t a_shape[] = {rank, columns};
    uint64_t b_shape[] = {output_rows, rank};
    h3_gpu_tensor *a = h3_weight_load_bf16(
        store, gpu, a_name, 2, a_shape, error, error_size);
    h3_gpu_tensor *b = h3_weight_load_bf16(
        store, gpu, b_name, 2, b_shape, error, error_size);
    int ok = a && b && h3_gpu_begin(gpu);
    if (ok && grouped_stream == -2)
        ok = h3_gpu_lora_merge_swap_halves_bf16(
            gpu, weight, a, b, columns, output_rows / 2, rank, 1.0f);
    else if (ok && grouped_stream >= 0)
        ok = h3_gpu_lora_merge_grouped_qkv_bf16(
            gpu, weight, a, b, columns, 56, 128,
            (uint32_t)grouped_stream, rank, 1.0f);
    else if (ok)
        ok = h3_gpu_lora_merge_bf16(
            gpu, weight, a, b, base_rows, columns, output_offset,
            output_rows, rank, 1.0f);
    if (ok) ok = h3_gpu_submit(gpu);
    h3_gpu_tensor_free(a);
    h3_gpu_tensor_free(b);
    if (!ok && (!error || !error_size || !error[0]))
        fail(error, error_size, "cannot merge VDN LoRA %s: %s", target,
             h3_gpu_error(gpu));
    return ok;
}

int h3_vdn_merge_target_bf16(
    const h3_vdn *vdn, h3_gpu *gpu, h3_gpu_tensor *weight,
    uint32_t base_rows, uint32_t columns, const char *target,
    uint32_t output_offset, uint32_t output_rows,
    char *error, size_t error_size) {
    if (!vdn || !gpu || !weight || !target ||
        output_offset > base_rows || output_rows > base_rows - output_offset) {
        fail(error, error_size, "invalid VDN LoRA merge request");
        return 0;
    }
    int found = 0;
    if (!merge_from(vdn->adapter_default, "default", gpu, weight,
                    base_rows, columns, target, output_offset, output_rows,
                    -1, &found, error, error_size) ||
        !merge_from(vdn->adapter_turbo, "turbo", gpu, weight,
                    base_rows, columns, target, output_offset, output_rows,
                    -1, &found, error, error_size)) return 0;
    if (!found) {
        fail(error, error_size, "required VDN adapter target is absent: %s",
             target);
        return 0;
    }
    return 1;
}

static int merge_grouped_target(const h3_vdn *vdn, h3_gpu *gpu,
                                h3_gpu_tensor *weight, uint32_t columns,
                                const char *target, int stream,
                                char *error, size_t error_size) {
    int found = 0;
    if (!merge_from(vdn->adapter_default, "default", gpu, weight,
                    56 * 128 * 3, columns, target, 0, 56 * 128,
                    stream, &found, error, error_size) ||
        !merge_from(vdn->adapter_turbo, "turbo", gpu, weight,
                    56 * 128 * 3, columns, target, 0, 56 * 128,
                    stream, &found, error, error_size)) return 0;
    if (!found) {
        fail(error, error_size, "required VDN adapter target is absent: %s",
             target);
        return 0;
    }
    return 1;
}

h3_gpu_tensor *h3_vdn_adapt_grouped_qkv_bf16(
    const h3_vdn *vdn, h3_gpu *gpu, const h3_gpu_tensor *base,
    uint32_t columns, uint32_t heads, uint32_t head_dim,
    const char *q_target, const char *k_target, const char *v_target,
    char *error, size_t error_size) {
    if (!vdn || !gpu || !base || heads != 56 || head_dim != 128 ||
        !columns || !q_target || !k_target || !v_target) {
        fail(error, error_size, "invalid VDN grouped QKV adapter request");
        return NULL;
    }
    uint32_t base_rows = heads * head_dim * 3;
    size_t elements = (size_t)base_rows * columns;
    h3_gpu_tensor *copy = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!copy || !h3_gpu_begin(gpu) ||
        !h3_gpu_copy_bf16(gpu, copy, 0, base, 0, elements) ||
        !h3_gpu_submit(gpu) ||
        !merge_grouped_target(vdn, gpu, copy, columns, q_target, 0,
                              error, error_size) ||
        !merge_grouped_target(vdn, gpu, copy, columns, k_target, 1,
                              error, error_size) ||
        !merge_grouped_target(vdn, gpu, copy, columns, v_target, 2,
                              error, error_size)) {
        if (copy && (!error || !error_size || !error[0]))
            fail(error, error_size, "cannot adapt VDN grouped QKV: %s",
                 h3_gpu_error(gpu));
        h3_gpu_tensor_free(copy);
        return NULL;
    }
    return copy;
}

h3_gpu_tensor *h3_vdn_adapt_swiglu_bf16(
    const h3_vdn *vdn, h3_gpu *gpu, const h3_gpu_tensor *base,
    uint32_t columns, uint32_t half_rows, const char *target,
    char *error, size_t error_size) {
    if (!vdn || !gpu || !base || !columns || !half_rows || !target ||
        half_rows > UINT32_MAX / 2) {
        fail(error, error_size, "invalid VDN SwiGLU adapter request");
        return NULL;
    }
    uint32_t rows = half_rows * 2;
    size_t elements = (size_t)rows * columns;
    h3_gpu_tensor *copy = h3_gpu_tensor_new_bf16(gpu, elements);
    int found = 0;
    if (!copy || !h3_gpu_begin(gpu) ||
        !h3_gpu_copy_bf16(gpu, copy, 0, base, 0, elements) ||
        !h3_gpu_submit(gpu) ||
        !merge_from(vdn->adapter_default, "default", gpu, copy,
                    rows, columns, target, 0, rows, -2, &found,
                    error, error_size) ||
        !merge_from(vdn->adapter_turbo, "turbo", gpu, copy,
                    rows, columns, target, 0, rows, -2, &found,
                    error, error_size) || !found) {
        if (!found && (!error || !error_size || !error[0]))
            fail(error, error_size,
                 "required VDN SwiGLU adapter target is absent: %s", target);
        h3_gpu_tensor_free(copy);
        return NULL;
    }
    return copy;
}

h3_gpu_tensor *h3_vdn_adapt_weight_bf16(
    const h3_vdn *vdn, h3_gpu *gpu, const h3_gpu_tensor *base,
    uint32_t base_rows, uint32_t columns, const char *target,
    uint32_t output_offset, uint32_t output_rows,
    char *error, size_t error_size) {
    if (!vdn || !gpu || !base || !base_rows || !columns) return NULL;
    size_t elements = (size_t)base_rows * columns;
    h3_gpu_tensor *copy = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!copy || !h3_gpu_begin(gpu) ||
        !h3_gpu_copy_bf16(gpu, copy, 0, base, 0, elements) ||
        !h3_gpu_submit(gpu) ||
        !h3_vdn_merge_target_bf16(
            vdn, gpu, copy, base_rows, columns, target, output_offset,
            output_rows, error, error_size)) {
        if (copy && (!error || !error_size || !error[0]))
            fail(error, error_size, "cannot copy VDN base weight: %s",
                 h3_gpu_error(gpu));
        h3_gpu_tensor_free(copy);
        return NULL;
    }
    return copy;
}
