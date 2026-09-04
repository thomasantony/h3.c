#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_TENSORS = 64 };

typedef struct {
    h3_gpu *gpu;
    h3_gpu_tensor *owned[MAX_TENSORS];
    size_t count;
} test_context;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_vdn_gpu.c: %s\n", message);
    exit(1);
}

static h3_gpu_tensor *own(test_context *test, h3_gpu_tensor *tensor) {
    if (!tensor || test->count == MAX_TENSORS) die("tensor allocation failed");
    test->owned[test->count++] = tensor;
    return tensor;
}

static void gpu_ok(test_context *test, int ok, const char *operation) {
    if (!ok) {
        fprintf(stderr, "FAIL tests/test_vdn_gpu.c: %s: %s\n", operation,
                h3_gpu_error(test->gpu));
        exit(1);
    }
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)((bits + rounding) >> 16);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float round_bf16(float value) {
    return bf16_to_f32(f32_to_bf16(value));
}

static h3_gpu_tensor *upload_bf16(test_context *test, const float *values,
                                  size_t count) {
    uint16_t *packed = malloc(count * sizeof(*packed));
    if (!packed) die("host BF16 allocation failed");
    for (size_t index = 0; index < count; index++)
        packed[index] = f32_to_bf16(values[index]);
    h3_gpu_tensor *tensor = own(test, h3_gpu_tensor_from_bf16(
                                         test->gpu, packed, count));
    free(packed);
    return tensor;
}

static h3_gpu_tensor *upload_f32(test_context *test, const float *values,
                                 size_t count) {
    return own(test, h3_gpu_tensor_from_f32(test->gpu, values, count));
}

static h3_gpu_tensor *fresh_f32(test_context *test, size_t count) {
    return own(test, h3_gpu_tensor_new_f32(test->gpu, count));
}

static h3_gpu_tensor *fresh_bf16(test_context *test, size_t count) {
    return own(test, h3_gpu_tensor_new_bf16(test->gpu, count));
}

static void compare_f32(const h3_gpu_tensor *tensor, const float *expected,
                        size_t count, float tolerance, const char *label) {
    float *got = malloc(count * sizeof(*got));
    if (!got || !h3_gpu_tensor_read_f32(tensor, got, count))
        die("cannot read F32 result");
    float maximum = 0.0f;
    for (size_t index = 0; index < count; index++) {
        float error = fabsf(got[index] - expected[index]);
        if (!isfinite(got[index]) || error > maximum) maximum = error;
    }
    printf("VDN primitive %-18s max abs %.7g\n", label, maximum);
    if (maximum > tolerance) {
        for (size_t index = 0; index < count; index++)
            fprintf(stderr, "  %s[%zu] got %.9g want %.9g\n", label, index,
                    got[index], expected[index]);
    }
    free(got);
    if (maximum > tolerance) die(label);
}

static void read_f32(const h3_gpu_tensor *tensor, float *values, size_t count) {
    if (!h3_gpu_tensor_read_f32(tensor, values, count))
        die("cannot read F32 result");
}

static void compare_bf16(const h3_gpu_tensor *tensor, const float *expected,
                         size_t count, float tolerance, const char *label) {
    uint16_t *packed = malloc(count * sizeof(*packed));
    if (!packed || !h3_gpu_tensor_read_bf16(tensor, packed, count))
        die("cannot read BF16 result");
    float maximum = 0.0f;
    for (size_t index = 0; index < count; index++) {
        float got = bf16_to_f32(packed[index]);
        float error = fabsf(got - expected[index]);
        if (!isfinite(got) || error > maximum) maximum = error;
    }
    printf("VDN primitive %-18s max abs %.7g\n", label, maximum);
    free(packed);
    if (maximum > tolerance) die(label);
}

static void matmul_square(float *output, const float *left,
                          const float *right, int dim) {
    for (int row = 0; row < dim; row++)
        for (int column = 0; column < dim; column++) {
            float sum = 0.0f;
            for (int k = 0; k < dim; k++)
                sum += left[row * dim + k] * right[k * dim + column];
            output[row * dim + column] = sum;
        }
}

static void inverse_spd(float *output, const float *matrix, int dim) {
    float work[4][8] = {{0}};
    if (dim > 4) die("CPU inverse test dimension is too large");
    for (int row = 0; row < dim; row++)
        for (int column = 0; column < dim; column++) {
            work[row][column] = matrix[row * dim + column] +
                                (row == column ? 1.0f : 0.0f);
            work[row][dim + column] = row == column ? 1.0f : 0.0f;
        }
    for (int pivot = 0; pivot < dim; pivot++) {
        float divisor = work[pivot][pivot];
        if (fabsf(divisor) < 1e-8f) die("singular CPU reference matrix");
        for (int column = 0; column < dim * 2; column++)
            work[pivot][column] /= divisor;
        for (int row = 0; row < dim; row++) {
            if (row == pivot) continue;
            float scale = work[row][pivot];
            for (int column = 0; column < dim * 2; column++)
                work[row][column] -= scale * work[pivot][column];
        }
    }
    for (int row = 0; row < dim; row++)
        for (int column = 0; column < dim; column++)
            output[row * dim + column] = work[row][dim + column];
}

static void test_statistics_solve_scan(test_context *test) {
    enum { FRAMES = 2, TOKENS = 3, HEADS = 1, DIM = 4,
           FEATURES = FRAMES * TOKENS * HEADS * DIM,
           MATRICES = FRAMES * HEADS * DIM * DIM,
           SYSTEMS = MATRICES * 2 };
    float key_values[FEATURES], value_values[FEATURES];
    for (int index = 0; index < FEATURES; index++) {
        key_values[index] = 0.625f * sinf((float)(index + 1));
        value_values[index] = 0.75f * cosf((float)(index * 2 + 1));
    }
    const float beta_values[FRAMES * TOKENS * HEADS] = {
        -0.75f, 0.25f, 1.0f, -0.5f, 0.75f, -1.25f,
    };
    float alpha_values[FRAMES * HEADS * DIM];
    float text_values[HEADS * DIM * DIM];
    for (int index = 0; index < FRAMES * HEADS * DIM; index++)
        alpha_values[index] = 0.93f - 0.035f * (float)index;
    for (int index = 0; index < HEADS * DIM * DIM; index++)
        text_values[index] = index % (DIM + 1) == 0 ? 0.1f :
                            0.01f * (float)(index - 5);
    h3_gpu_tensor *key = upload_bf16(test, key_values, FEATURES);
    h3_gpu_tensor *value = upload_bf16(test, value_values, FEATURES);
    h3_gpu_tensor *beta = upload_bf16(test, beta_values,
                                      FRAMES * TOKENS * HEADS);
    h3_gpu_tensor *alpha = upload_f32(test, alpha_values,
                                      FRAMES * HEADS * DIM);
    h3_gpu_tensor *a = fresh_f32(test, MATRICES);
    h3_gpu_tensor *b = fresh_f32(test, MATRICES);
    h3_gpu_tensor *rhs = fresh_f32(test, SYSTEMS);
    h3_gpu_tensor *solution = fresh_f32(test, SYSTEMS);
    h3_gpu_tensor *text = upload_f32(test, text_values, HEADS * DIM * DIM);
    h3_gpu_tensor *prefix = fresh_f32(test, MATRICES);
    h3_gpu_tensor *suffix = fresh_f32(test, MATRICES);

    float expected_a[MATRICES] = {0};
    float expected_b[MATRICES] = {0};
    for (int frame = 0; frame < FRAMES; frame++)
        for (int token = 0; token < TOKENS; token++) {
            int feature = (frame * TOKENS + token) * DIM;
            float sigmoid = 1.0f / (1.0f + expf(
                -round_bf16(beta_values[frame * TOKENS + token])));
            sigmoid = round_bf16(sigmoid);
            for (int row = 0; row < DIM; row++)
                for (int column = 0; column < DIM; column++) {
                    float k_row = round_bf16(key_values[feature + row]);
                    float k_column = round_bf16(key_values[feature + column]);
                    float v_row = round_bf16(value_values[feature + row]);
                    expected_a[(frame * DIM + row) * DIM + column] +=
                        k_row * sigmoid * k_column;
                    expected_b[(frame * DIM + row) * DIM + column] +=
                        round_bf16(v_row * sigmoid) * k_column;
                }
        }
    /* The stats buffer carries A+A^T; the solve-prep kernel folds the 0.5
     * symmetrization into the same pass that adds identity. */
    for (int index = 0; index < MATRICES; index++) expected_a[index] *= 2.0f;

    gpu_ok(test, h3_gpu_begin(test->gpu), "begin statistics");
    gpu_ok(test, h3_gpu_vdn_statistics_f32(
                     test->gpu, a, b, key, value, beta,
                     FRAMES, TOKENS, HEADS, DIM), "frame statistics");
    gpu_ok(test, h3_gpu_submit(test->gpu), "submit statistics");
    compare_f32(a, expected_a, MATRICES, 0.005f, "statistics A");
    compare_f32(b, expected_b, MATRICES, 0.012f, "statistics B");

    float source_a[MATRICES], source_b[MATRICES];
    read_f32(a, source_a, MATRICES);
    read_f32(b, source_b, MATRICES);
    float expected_transition[MATRICES], expected_injection[MATRICES];
    for (int frame = 0; frame < FRAMES; frame++) {
        float symmetric_a[DIM * DIM], inverse[DIM * DIM];
        float injection[DIM * DIM];
        for (int index = 0; index < DIM * DIM; index++)
            symmetric_a[index] = 0.5f *
                                 source_a[frame * DIM * DIM + index];
        inverse_spd(inverse, symmetric_a, DIM);
        matmul_square(injection, source_b + frame * DIM * DIM, inverse, DIM);
        for (int row = 0; row < DIM; row++)
            for (int column = 0; column < DIM; column++) {
                expected_transition[(frame * DIM + row) * DIM + column] =
                    alpha_values[frame * DIM + row] *
                    inverse[row * DIM + column];
                expected_injection[(frame * DIM + row) * DIM + column] =
                    injection[row * DIM + column];
            }
    }
    gpu_ok(test, h3_gpu_begin(test->gpu), "begin solve");
    gpu_ok(test, h3_gpu_vdn_solve_f32(
                     test->gpu, a, b, rhs, solution, alpha,
                     FRAMES, HEADS, DIM), "joint solve");
    gpu_ok(test, h3_gpu_submit(test->gpu), "submit solve");
    compare_f32(b, expected_injection, MATRICES, 0.001f, "solve injection");
    float solved[SYSTEMS];
    read_f32(solution, solved, SYSTEMS);
    float got_transition[MATRICES];
    memcpy(got_transition, solved, sizeof(got_transition));
    for (int index = 0; index < MATRICES; index++)
        if (fabsf(got_transition[index] - expected_transition[index]) > 0.001f)
            die("solve transition");
    puts("VDN primitive solve transition   max abs < 0.001");

    float got_injection[MATRICES];
    read_f32(b, got_injection, MATRICES);
    float expected_prefix[MATRICES], expected_suffix[MATRICES];
    float state[DIM * DIM], next[DIM * DIM];
    memcpy(state, text_values, sizeof(state));
    for (int frame = 0; frame < FRAMES; frame++) {
        matmul_square(next, state,
                      got_transition + frame * DIM * DIM, DIM);
        for (int index = 0; index < DIM * DIM; index++)
            next[index] += got_injection[frame * DIM * DIM + index];
        memcpy(expected_prefix + frame * DIM * DIM, next, sizeof(next));
        memcpy(state, next, sizeof(state));
    }
    memcpy(state, text_values, sizeof(state));
    for (int frame = FRAMES - 1; frame >= 0; frame--) {
        matmul_square(next, state,
                      got_transition + frame * DIM * DIM, DIM);
        for (int index = 0; index < DIM * DIM; index++)
            next[index] += got_injection[frame * DIM * DIM + index];
        memcpy(expected_suffix + frame * DIM * DIM, next, sizeof(next));
        memcpy(state, next, sizeof(state));
    }
    gpu_ok(test, h3_gpu_begin(test->gpu), "begin scans");
    gpu_ok(test, h3_gpu_vdn_scan_f32(
                     test->gpu, prefix, suffix, b, solution, text,
                     FRAMES, HEADS, DIM), "bidirectional scans");
    gpu_ok(test, h3_gpu_submit(test->gpu), "submit scans");
    compare_f32(prefix, expected_prefix, MATRICES, 0.001f, "forward scan");
    compare_f32(suffix, expected_suffix, MATRICES, 0.001f, "reverse scan");
}

static void test_gather(test_context *test) {
    enum { FRAMES = 12, HEADS = 1, DIM = 1, COUNT = FRAMES };
    float prefix_values[COUNT], suffix_values[COUNT], alpha_values[COUNT];
    for (int frame = 0; frame < FRAMES; frame++) {
        prefix_values[frame] = 10.0f + (float)frame;
        suffix_values[frame] = 100.0f + (float)frame;
        alpha_values[frame] = 0.91f - 0.01f * (float)frame;
    }
    const float text_values[] = {5.0f};
    h3_gpu_tensor *prefix = upload_f32(test, prefix_values, COUNT);
    h3_gpu_tensor *suffix = upload_f32(test, suffix_values, COUNT);
    h3_gpu_tensor *alpha = upload_f32(test, alpha_values, COUNT);
    h3_gpu_tensor *text = upload_f32(test, text_values, 1);
    h3_gpu_tensor *state = fresh_bf16(test, COUNT);
    float expected[COUNT];
    for (int frame = 0; frame < FRAMES; frame++) {
        int original = frame + 1;
        int chunk = original / 5;
        int lo = (chunk - 1) * 5 - 1;
        int hi = (chunk + 2) * 5 - 2;
        int before_index = lo - 1;
        int after_index = hi + 1;
        int has_before = before_index >= 0;
        int has_after = after_index < FRAMES;
        float before = has_before ? prefix_values[before_index] : text_values[0];
        float after = has_after ? suffix_values[after_index] : text_values[0];
        int bridge_before = before_index + 1;
        if (bridge_before < 0) bridge_before = 0;
        int bridge_after = after_index;
        if (bridge_after > FRAMES) bridge_after = FRAMES;
        float before_log = 0.0f, after_log = 0.0f;
        for (int index = bridge_before; index <= frame; index++)
            before_log += logf(fmaxf(alpha_values[index], 1e-12f));
        for (int index = frame; index < bridge_after; index++)
            after_log += logf(fmaxf(alpha_values[index], 1e-12f));
        expected[frame] = before * expf(before_log) + after * expf(after_log);
    }
    gpu_ok(test, h3_gpu_begin(test->gpu), "begin gather");
    gpu_ok(test, h3_gpu_vdn_gather_state_bf16(
                     test->gpu, state, prefix, suffix, alpha, text,
                     FRAMES, HEADS, DIM), "state gather");
    gpu_ok(test, h3_gpu_submit(test->gpu), "submit gather");
    compare_bf16(state, expected, COUNT, 0.26f, "alpha bridge");
}

static void test_h3_sized_solve(test_context *test) {
    enum { DIM = 128, TOKENS = 48, MATRIX = DIM * DIM, SYSTEM = MATRIX * 2 };
    float *a_values = calloc(MATRIX, sizeof(*a_values));
    float *b_values = malloc(MATRIX * sizeof(*b_values));
    float *alpha_values = malloc(DIM * sizeof(*alpha_values));
    double *factor = calloc(MATRIX, sizeof(*factor));
    double *inverse = calloc(MATRIX, sizeof(*inverse));
    float *expected_transition = malloc(MATRIX * sizeof(*expected_transition));
    float *expected_injection = malloc(MATRIX * sizeof(*expected_injection));
    if (!a_values || !b_values || !alpha_values || !factor || !inverse ||
        !expected_transition || !expected_injection)
        die("H3-sized solve allocation failed");
    for (int token = 0; token < TOKENS; token++) {
        float key[DIM];
        double square = 0.0;
        for (int channel = 0; channel < DIM; channel++) {
            key[channel] = sinf(0.013f * (float)(channel + 3) +
                                 0.11f * (float)token) +
                           0.15f * cosf(0.07f * (float)channel -
                                        0.17f * (float)token);
            square += (double)key[channel] * key[channel];
        }
        float scale = (float)(1.0 / sqrt(square));
        float beta = 0.2f + 0.6f / (1.0f + expf(-0.1f * (float)(token - 20)));
        for (int row = 0; row < DIM; row++)
            for (int column = 0; column < DIM; column++)
                a_values[row * DIM + column] +=
                    2.0f * beta * key[row] * scale * key[column] * scale;
    }
    for (int row = 0; row < DIM; row++) {
        alpha_values[row] = 0.7f + 0.25f *
                            (float)(row % 17) / 16.0f;
        for (int column = 0; column < DIM; column++) {
            b_values[row * DIM + column] =
                0.02f * sinf(0.09f * (float)(row * 3 + column * 5));
            factor[row * DIM + column] =
                0.5 * (double)a_values[row * DIM + column] +
                (row == column ? 1.0 : 0.0);
        }
    }
    for (int row = 0; row < DIM; row++) {
        for (int column = 0; column <= row; column++) {
            double value = factor[row * DIM + column];
            for (int k = 0; k < column; k++)
                value -= factor[row * DIM + k] * factor[column * DIM + k];
            factor[row * DIM + column] = row == column ? sqrt(value) :
                value / factor[column * DIM + column];
        }
    }
    double vector[DIM];
    for (int column = 0; column < DIM; column++) {
        for (int row = 0; row < DIM; row++) {
            double value = row == column ? 1.0 : 0.0;
            for (int k = 0; k < row; k++)
                value -= factor[row * DIM + k] * vector[k];
            vector[row] = value / factor[row * DIM + row];
        }
        for (int row = DIM - 1; row >= 0; row--) {
            double value = vector[row];
            for (int k = row + 1; k < DIM; k++)
                value -= factor[k * DIM + row] *
                         inverse[k * DIM + column];
            inverse[row * DIM + column] =
                value / factor[row * DIM + row];
        }
    }
    for (int row = 0; row < DIM; row++)
        for (int column = 0; column < DIM; column++) {
            double injection = 0.0;
            for (int k = 0; k < DIM; k++)
                injection += (double)b_values[row * DIM + k] *
                             inverse[k * DIM + column];
            expected_transition[row * DIM + column] =
                alpha_values[row] * (float)inverse[row * DIM + column];
            expected_injection[row * DIM + column] = (float)injection;
        }
    h3_gpu_tensor *a = upload_f32(test, a_values, MATRIX);
    h3_gpu_tensor *b = upload_f32(test, b_values, MATRIX);
    h3_gpu_tensor *alpha = upload_f32(test, alpha_values, DIM);
    h3_gpu_tensor *rhs = fresh_f32(test, SYSTEM);
    h3_gpu_tensor *transition = fresh_f32(test, SYSTEM);
    gpu_ok(test, h3_gpu_begin(test->gpu), "begin H3-sized solve");
    gpu_ok(test, h3_gpu_vdn_solve_f32(
                     test->gpu, a, b, rhs, transition, alpha,
                     1, 1, DIM), "H3-sized solve");
    gpu_ok(test, h3_gpu_submit(test->gpu), "submit H3-sized solve");
    compare_f32(transition, expected_transition, MATRIX, 0.00002f,
                "H3 solve transition");
    compare_f32(b, expected_injection, MATRIX, 0.00002f,
                "H3 solve injection");
    free(a_values); free(b_values); free(alpha_values); free(factor);
    free(inverse); free(expected_transition); free(expected_injection);
}

static void test_readout_and_lora(test_context *test) {
    enum { FRAMES = 2, TOKENS = 3, HEADS = 1, DIM = 2,
           FEATURES = FRAMES * TOKENS * HEADS * DIM,
           MATRICES = FRAMES * HEADS * DIM * DIM };
    const float query_values[FEATURES] = {
        0.5f, -0.25f, 0.75f, 0.125f, -0.5f, 0.625f,
        0.25f, 0.875f, -0.75f, 0.5f, 0.125f, -0.375f,
    };
    const float state_values[MATRICES] = {
        0.5f, -0.25f, 0.75f, 0.125f,
        -0.5f, 0.25f, 0.625f, 0.875f,
    };
    h3_gpu_tensor *query = upload_bf16(test, query_values, FEATURES);
    h3_gpu_tensor *state = upload_bf16(test, state_values, MATRICES);
    h3_gpu_tensor *output = fresh_bf16(test, FEATURES);
    float expected[FEATURES];
    for (int frame = 0; frame < FRAMES; frame++)
        for (int token = 0; token < TOKENS; token++)
            for (int value_channel = 0; value_channel < DIM; value_channel++) {
                float sum = 0.0f;
                for (int key_channel = 0; key_channel < DIM; key_channel++)
                    sum += round_bf16(query_values[
                               (frame * TOKENS + token) * DIM + key_channel]) *
                           round_bf16(state_values[
                               frame * 4 + value_channel * DIM + key_channel]);
                expected[(frame * TOKENS + token) * DIM + value_channel] = sum;
            }
    gpu_ok(test, h3_gpu_begin(test->gpu), "begin readout");
    gpu_ok(test, h3_gpu_vdn_readout_bf16(
                     test->gpu, output, query, state,
                     FRAMES, TOKENS, HEADS, DIM), "state readout");
    gpu_ok(test, h3_gpu_submit(test->gpu), "submit readout");
    compare_bf16(output, expected, FEATURES, 0.01f, "BF16 readout");

    enum { COLUMNS = 2, QKV_HEADS = 2, QKV_DIM = 2,
           QKV_ROWS = QKV_HEADS * QKV_DIM * 3 };
    float zero_qkv[QKV_ROWS * COLUMNS];
    memset(zero_qkv, 0, sizeof(zero_qkv));
    const float lora_a[] = {0.5f, -1.0f};
    const float lora_b[] = {1.0f, 2.0f, 3.0f, 4.0f};
    h3_gpu_tensor *qkv = upload_bf16(test, zero_qkv,
                                     QKV_ROWS * COLUMNS);
    h3_gpu_tensor *a = upload_bf16(test, lora_a, 2);
    h3_gpu_tensor *b = upload_bf16(test, lora_b, 4);
    float expected_qkv[QKV_ROWS * COLUMNS];
    memset(expected_qkv, 0, sizeof(expected_qkv));
    for (int row = 0; row < QKV_HEADS * QKV_DIM; row++) {
        int head = row / QKV_DIM;
        int channel = row % QKV_DIM;
        int destination = (head * 3 + 1) * QKV_DIM + channel;
        for (int column = 0; column < COLUMNS; column++)
            expected_qkv[destination * COLUMNS + column] =
                lora_b[row] * lora_a[column];
    }
    gpu_ok(test, h3_gpu_begin(test->gpu), "begin grouped LoRA");
    gpu_ok(test, h3_gpu_lora_merge_grouped_qkv_bf16(
                     test->gpu, qkv, a, b, COLUMNS, QKV_HEADS, QKV_DIM,
                     1, 1, 1.0f), "grouped QKV LoRA");
    gpu_ok(test, h3_gpu_submit(test->gpu), "submit grouped LoRA");
    compare_bf16(qkv, expected_qkv, QKV_ROWS * COLUMNS, 0.001f,
                 "grouped QKV LoRA");

    enum { HALF_ROWS = 3, SWIGLU_ROWS = HALF_ROWS * 2 };
    float zero_swiglu[SWIGLU_ROWS * COLUMNS];
    memset(zero_swiglu, 0, sizeof(zero_swiglu));
    const float swiglu_b[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    h3_gpu_tensor *swiglu = upload_bf16(test, zero_swiglu,
                                        SWIGLU_ROWS * COLUMNS);
    h3_gpu_tensor *swiglu_b_tensor = upload_bf16(test, swiglu_b, SWIGLU_ROWS);
    float expected_swiglu[SWIGLU_ROWS * COLUMNS];
    memset(expected_swiglu, 0, sizeof(expected_swiglu));
    for (int row = 0; row < SWIGLU_ROWS; row++) {
        int destination = row < HALF_ROWS ? row + HALF_ROWS : row - HALF_ROWS;
        for (int column = 0; column < COLUMNS; column++)
            expected_swiglu[destination * COLUMNS + column] =
                swiglu_b[row] * lora_a[column];
    }
    gpu_ok(test, h3_gpu_begin(test->gpu), "begin SwiGLU LoRA");
    gpu_ok(test, h3_gpu_lora_merge_swap_halves_bf16(
                     test->gpu, swiglu, a, swiglu_b_tensor,
                     COLUMNS, HALF_ROWS, 1, 1.0f), "SwiGLU LoRA");
    gpu_ok(test, h3_gpu_submit(test->gpu), "submit SwiGLU LoRA");
    compare_bf16(swiglu, expected_swiglu, SWIGLU_ROWS * COLUMNS, 0.001f,
                 "SwiGLU LoRA layout");
}

int main(void) {
    test_context test = {0};
    char error[512];
    test.gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!test.gpu) die(error);
    test_statistics_solve_scan(&test);
    test_gather(&test);
    test_h3_sized_solve(&test);
    test_readout_and_lora(&test);
    for (size_t index = 0; index < test.count; index++)
        h3_gpu_tensor_free(test.owned[index]);
    h3_gpu_free(test.gpu);
    puts("ok: VDN Metal primitives and checkpoint layouts");
    return 0;
}
