#include "CImg.h"  // CImg library for image handling
#include <CL/opencl.hpp>  // OpenCL C++ bindings
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <limits>
#include <cmath>

using namespace cimg_library;
using namespace std;

// Returns kernel execution time in milliseconds
double getEventDurationMS(const cl::Event& event) {
    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    return (end - start) * 1e-6;
}

// Finds first and last non-zero bins in histogram
pair<int, int> find_min_max_bin(const vector<cl_uint>& hist) {
    int min_bin = 0, max_bin = hist.size() - 1;
    while (min_bin < hist.size() && hist[min_bin] == 0) ++min_bin;
    while (max_bin >= 0 && hist[max_bin] == 0) --max_bin;
    return { min_bin, max_bin };
}

int main(int argc, char* argv[]) {
    // Default to GPU (0), but allow override
    int platform_type = 0;  // 0 = GPU, 1 = CPU
    if (argc == 3 && string(argv[1]) == "-p") {
        platform_type = atoi(argv[2]);  // 0 or 1
    }
    auto start_total = chrono::high_resolution_clock::now();  // Start timing

    // Load input image (grayscale or RGB)
    CImg<unsigned char> image("mdr16.ppm");
    int width = image.width(), height = image.height(), spectrum = image.spectrum();
    bool is_colour = (spectrum == 3);

    const int num_bins = 256;
    int size = width * height;
    vector<unsigned char> output(size * (is_colour ? 3 : 1));  // Output buffer (RGB or grayscale)

    // Generate luminance vector (convert RGB to grayscale if needed)
    vector<unsigned char> luminance(size);
    if (is_colour) {
        cimg_forXY(image, x, y) {
            unsigned char R = image(x, y, 0, 0);
            unsigned char G = image(x, y, 0, 1);
            unsigned char B = image(x, y, 0, 2);
            luminance[y * width + x] = static_cast<unsigned char>(0.299 * R + 0.587 * G + 0.114 * B);
        }
    }
    else {
        cimg_forXY(image, x, y) {
            luminance[y * width + x] = image(x, y);  // Already grayscale
        }
    }

    // OpenCL setup: platform, device, context, and program
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();

    vector<cl::Device> devices;
    platform.getDevices(platform_type == 0 ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, &devices);
    cl::Device device = devices.front();

    cl::Context context({ device });
    cl::Program::Sources sources;

    ifstream kernel_file("my_kernels.cl");  // Load OpenCL kernels
    string kernel_code((istreambuf_iterator<char>(kernel_file)), istreambuf_iterator<char>());
    sources.push_back({ kernel_code.c_str(), kernel_code.length() });

    cl::Program program(context, sources);
    program.build({ device });  // Compile kernels
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);  // Enable profiling

    // Allocate device buffers
    cl::Buffer buf_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * size, luminance.data());
    cl::Buffer buf_output(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * size);
    cl::Buffer buf_hist(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * num_bins);
    cl::Buffer buf_scan(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * num_bins);
    cl::Buffer buf_lut(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * num_bins);

    vector<cl_uint> hist_init(num_bins, 0);  // Zero-initialised histogram

    // Transfer histogram buffer to GPU and time it
    auto start_mem = chrono::high_resolution_clock::now();
    queue.enqueueWriteBuffer(buf_hist, CL_TRUE, 0, sizeof(cl_uint) * num_bins, hist_init.data());
    auto end_mem = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> mem_time = end_mem - start_mem;

    // Declare profiling events
    cl::Event evt_hist, evt_scan, evt_norm, evt_apply;

    // Run histogram kernel to populate histogram bins
    cl::Kernel histogram(program, "histogram_kernel");
    histogram.setArg(0, buf_input);
    histogram.setArg(1, buf_hist);
    histogram.setArg(2, size);
    histogram.setArg(3, num_bins);
    queue.enqueueNDRangeKernel(histogram, cl::NullRange, cl::NDRange(size), cl::NullRange, nullptr, &evt_hist);

    // Run scan kernel using Hillis-Steele prefix sum (for cumulative histogram)
    cl::Kernel scan(program, "scan_kernel");
    scan.setArg(0, buf_hist);
    scan.setArg(1, buf_scan);
    scan.setArg(2, num_bins);
    queue.enqueueNDRangeKernel(scan, cl::NullRange, cl::NDRange(num_bins), cl::NullRange, nullptr, &evt_scan);

    // Read scan result to CPU and determine min/max bin
    vector<cl_uint> scan_result(num_bins);
    queue.enqueueReadBuffer(buf_scan, CL_TRUE, 0, sizeof(cl_uint) * num_bins, scan_result.data());
    pair<int, int> minmax = find_min_max_bin(scan_result);
    int min_bin = minmax.first;
    int max_bin = minmax.second;
    vector<cl_uint> hist_result(num_bins);

    // Run normalisation kernel to compute LUT from cumulative histogram
    cl::Kernel normalise(program, "normalise_kernel");
    normalise.setArg(0, buf_scan);
    normalise.setArg(1, buf_lut);
    normalise.setArg(2, num_bins);
    normalise.setArg(3, size);
    normalise.setArg(4, cl_int(min_bin));
    normalise.setArg(5, cl_int(max_bin));
    queue.enqueueNDRangeKernel(normalise, cl::NullRange, cl::NDRange(num_bins), cl::NullRange, nullptr, &evt_norm);

    // Apply LUT to image using equalisation values
    cl::Kernel apply(program, "apply_lut_kernel");
    apply.setArg(0, buf_input);
    apply.setArg(1, buf_output);
    apply.setArg(2, buf_lut);
    apply.setArg(3, size);
    apply.setArg(4, num_bins);
    queue.enqueueNDRangeKernel(apply, cl::NullRange, cl::NDRange(size), cl::NullRange, nullptr, &evt_apply);

    // Read back equalised image
    cl::Event evt_read;
    vector<unsigned char> equalised_luminance(size);
    queue.enqueueReadBuffer(buf_output, CL_FALSE, 0, sizeof(unsigned char) * size, equalised_luminance.data(), nullptr, &evt_read);
    queue.finish();  // Wait for all GPU tasks to complete

    // Output kernel timings
    cout << "\n=== Execution Times ===\n";
    cout << "Memory Transfer (init): " << mem_time.count() << " ms\n";
    cout << "Histogram kernel:       " << getEventDurationMS(evt_hist) << " ms\n";
    cout << "Scan kernel:            " << getEventDurationMS(evt_scan) << " ms\n";
    cout << "Normalise kernel:       " << getEventDurationMS(evt_norm) << " ms\n";
    cout << "Apply LUT kernel:       " << getEventDurationMS(evt_apply) << " ms\n";
    cout << "Read buffer (output):   " << getEventDurationMS(evt_read) << " ms\n";

    // Reconstruct final image: apply luminance changes to RGB if colour
    if (is_colour) {
        cimg_forXY(image, x, y) {
            int idx = y * width + x;
            float Y_old = static_cast<float>(luminance[idx]);
            float Y_new = static_cast<float>(equalised_luminance[idx]);
            float scale = (Y_old > 0.0f) ? (Y_new / Y_old) : 0.0f;

            for (int c = 0; c < 3; ++c) {
                float val = static_cast<float>(image(x, y, 0, c));
                float adjusted = val * scale;
                output[3 * idx + c] = static_cast<unsigned char>(fmin(255.0f, fmax(0.0f, adjusted)));
            }
        }

        CImg<unsigned char> result_image(output.data(), width, height, 1, 3);
        result_image.save("output_equalised.ppm");  // Save colour output
    }
    else {
        std::copy(equalised_luminance.begin(), equalised_luminance.end(), output.begin());
        CImg<unsigned char> result_image(output.data(), width, height, 1, 1);
        result_image.save("output_equalised.pgm");  // Save grayscale output
    }

    auto end_total = chrono::high_resolution_clock::now();  // End timing
    chrono::duration<double, milli> total_time = end_total - start_total;

    cout << fixed << setprecision(3);
    cout << "\nHistogram Equalisation complete. Output saved to output_equalised." << (is_colour ? "ppm" : "pgm") << endl;
    cout << "Total Program Time:      " << total_time.count() << " ms" << endl;

    return 0;
}
