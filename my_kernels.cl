// ========== my_kernels.cl ==========

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef unsigned char uchar;
typedef unsigned int uint;

// 1. Compute histogram (global memory version)
kernel void histogram_kernel(__global uchar* input, __global uint* hist, int size) {
    int id = get_global_id(0);
    if (id < size) {
        atomic_inc(&hist[input[id]]);
    }
}

// 2. Inclusive scan using Hillis-Steele
kernel void scan_kernel(__global uint* input, __global uint* output, int size) {
    int id = get_global_id(0);
    if (id >= size) return;
    output[id] = input[id];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int offset = 1; offset < size; offset <<= 1) {
        uint val = 0;
        if (id >= offset)
            val = output[id - offset];
        barrier(CLK_GLOBAL_MEM_FENCE);
        output[id] += val;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

// 3. Normalize the cumulative histogram into LUT
kernel void normalize_kernel(__global uint* cum_hist, __global uchar* lut, int size, int total_pixels) {
    int id = get_global_id(0);
    if (id >= size) return;

    uint min_val = cum_hist[0];
    float norm = (float)(cum_hist[id] - min_val) / (float)(total_pixels - min_val);
    norm = fmax(0.0f, fmin(norm, 1.0f));
    lut[id] = (uchar)(255.0f * norm + 0.5f);
}

// 4. Map the LUT to the image (back-projection)
kernel void apply_lut_kernel(__global uchar* input, __global uchar* output, __global uchar* lut, int size) {
    int id = get_global_id(0);
    if (id < size) {
        output[id] = lut[input[id]];
    }
}
