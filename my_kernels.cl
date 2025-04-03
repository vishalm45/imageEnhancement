// ========== my_kernels.cl ==========

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef unsigned char uchar;
typedef unsigned int uint;

// 1. Compute histogram (global memory version)
kernel void histogram_kernel(__global uchar* input, __global uint* hist, int size, int num_bins) {
    int id = get_global_id(0);
    if (id < size) {
        uchar pixel = input[id];
        if (pixel < num_bins)
            atomic_inc(&hist[pixel]);
    }
}

// 1b. Compute histogram for a single RGB channel
kernel void histogram_kernel_channel(__global uchar* input, __global uint* hist, int pixel_count, int channel, int num_bins) {
    int id = get_global_id(0);
    if (id < pixel_count) {
        uchar val = input[3 * id + channel];
        if (val < num_bins)
            atomic_inc(&hist[val]);
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

// 3. Normalise the cumulative histogram into LUT using proper cumulative stretching
kernel void normalise_kernel(__global uint* cum_hist, __global uchar* lut, int size, int total_pixels, int min_bin, int max_bin) {
    int id = get_global_id(0);
    if (id >= size) return;

    uint min_val = cum_hist[min_bin];
    uint max_val = cum_hist[max_bin];
    uint val = cum_hist[id];

    if (max_val == min_val) {
        lut[id] = 0;
    }
    else {
        float norm = (float)(val - min_val) / (float)(max_val - min_val);
        norm = fmax(0.0f, fmin(norm, 1.0f));
        lut[id] = (uchar)(255.0f * norm + 0.5f);
    }
}

// 4. Map the LUT to the grayscale image (back-projection)
kernel void apply_lut_kernel(__global uchar* input, __global uchar* output, __global uchar* lut, int size, int num_bins) {
    int id = get_global_id(0);
    if (id < size) {
        uchar val = input[id];
        if (val < num_bins)
            output[id] = lut[val];
    }
}

// 5. Apply 3 LUTs to RGB image
kernel void apply_lut_kernel_color(__global uchar* input, __global uchar* output,
    __global uchar* lut_r, __global uchar* lut_g, __global uchar* lut_b,
    int pixel_count) {
    int id = get_global_id(0);
    if (id < pixel_count) {
        uchar r = input[3 * id + 0];
        uchar g = input[3 * id + 1];
        uchar b = input[3 * id + 2];
        output[3 * id + 0] = lut_r[r];
        output[3 * id + 1] = lut_g[g];
        output[3 * id + 2] = lut_b[b];
    }
}
