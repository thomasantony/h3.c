#include "h3_gpu.h"
#include "h3_vdn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_vdn_checkpoint.c: %s\n", message);
    exit(1);
}

static void check_checkpoint(h3_gpu *gpu, const char *path,
                             int expected_steps) {
    char error[512];
    int steps = h3_vdn_recommended_steps(path, error, sizeof(error));
    if (steps != expected_steps) {
        fprintf(stderr, "FAIL tests/test_vdn_checkpoint.c: %s recommends %d: %s\n",
                path, steps, error);
        exit(1);
    }
    h3_vdn *vdn = h3_vdn_open(path, error, sizeof(error));
    if (!vdn) die(error);
    for (unsigned block = 0; block < 50; block += 49) {
        h3_vdn_block weights;
        memset(&weights, 0, sizeof(weights));
        if (!h3_vdn_load_block(vdn, gpu, block, &weights,
                               error, sizeof(error))) die(error);
        h3_vdn_free_block(&weights);
    }
    printf("ok: direct VDN checkpoint load %s (%d steps)\n", path, steps);
    h3_vdn_free(vdn);
}

int main(int argc, char **argv) {
    const char *stage_b = argc > 1 ? argv[1] :
        "../vdn-minimax-h3/ckpts/stage-b-step-2000";
    const char *stage_dmd = argc > 2 ? argv[2] :
        "../vdn-minimax-h3/ckpts/stage-dmd-step-250";
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);
    check_checkpoint(gpu, stage_b, 50);
    check_checkpoint(gpu, stage_dmd, 8);
    h3_gpu_free(gpu);
    return 0;
}
