// ========== my_kernels.cl ==========

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef unsigned char uchar;
typedef unsigned int uint;

// 1. Builds histogram using atomic increments in global memory
kernel void histogram_kernel(__global uchar* input, __global uint* hist, int size, int num_bins) {
    int id = get_global_id(0);
    if (id < size) {
        atomic_inc(&hist[input[id]]);  // Thread-safe increment per bin
    }
}

// 2. Inclusive prefix sum using Hillis-Steele algorithm (for cumulative histogram)
kernel void scan_kernel(__global uint* input, __global uint* output, int size) {
    int id = get_global_id(0);
    if (id >= size) return;

    output[id] = input[id];  // Initial copy
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Hillis-Steele parallel scan loop (inclusive)
    for (int offset = 1; offset < size; offset <<= 1) {
        if (id < size) {
            uint val = 0;
            if (id >= offset) {
                val = output[id - offset];
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            output[id] += val;
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

// 3. Normalises cumulative histogram into [0, 255] lookup table (LUT)
kernel void normalise_kernel(__global uint* cum_hist, __global uchar* lut, int num_bins, int total_pixels, int min_bin, int max_bin) {
    int id = get_global_id(0);
    if (id >= num_bins) return;

    uint cdf_min = cum_hist[min_bin];
    uint cdf_max = cum_hist[max_bin];
    uint cdf_val = cum_hist[id];

    float norm;
    if (cdf_max > cdf_min) {
        norm = (float)(cdf_val - cdf_min) / (float)(cdf_max - cdf_min); // Normalise to [0, 1]
    } else {
        norm = 0.0f; // Or handle differently based on your desired behavior
    }
    norm = fmax(0.0f, fmin(norm, 1.0f)); // Clamp
    lut[id] = (uchar)(255.0f * norm + 0.5f); // Convert to 8-bit
}

// 4. Applies LUT to greyscale image (back-projection step)
kernel void apply_lut_kernel(__global uchar* input,
    __global uchar* output,
    __global uchar* lut,
    int size) {
    int id = get_global_id(0);
    if (id < size) {
        output[id] = lut[input[id]]; // Replace intensity using LUT
    }
}

// 5. Applies 3 separate LUTs to RGB image (for independent channel equalisation)
kernel void apply_lut_kernel_colour(__global uchar* input, __global uchar* output,
    __global uchar* lut_r, __global uchar* lut_g, __global uchar* lut_b,
    int pixel_count) {
    int id = get_global_id(0);
    if (id < pixel_count) {
        uchar r = input[3 * id + 0];
        uchar g = input[3 * id + 1];
        uchar b = input[3 * id + 2];
        output[3 * id + 0] = lut_r[r]; // Red channel LUT
        output[3 * id + 1] = lut_g[g]; // Green channel LUT
        output[3 * id + 2] = lut_b[b]; // Blue channel LUT
    }
}
