#include <iostream>
#include <vector>

#include "Utils.h"   // Assuming this header has any utility functions you need
#include "CImg.h"    // For image handling with CImg library

#include <CL/opencl.hpp>  // OpenCL C++ bindings

using namespace std;
using namespace cimg_library; // This allows you to use CImg without the prefix

// Returns kernel execution time in milliseconds using cl::Event
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
    } else {
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

    // Use ifstream to load kernel code from file
    ifstream kernel_file("my_kernels.cl");
    string kernel_code((istreambuf_iterator<char>(kernel_file)), istreambuf_iterator<char>());
    sources.push_back({ kernel_code.c_str(), kernel_code.length() });

    cl::Program program(context, sources);
    program.build({ device });  // Compile kernels
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);  // Enable profiling

    // Allocate device buffers
    cl::Buffer buf_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * size, luminance.data());
    cl::Buffer buf_output(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * size * (is_colour ? 3 : 1));  // Adjusted for color images
    cl::Buffer buf_hist(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * num_bins);
    cl::Buffer buf_scan(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * num_bins);
    cl::Buffer buf_lut(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * num_bins);

    vector<cl_uint> hist_init(num_bins, 0);  // Zero-initialised histogram

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

    // Create LUTs for RGB channels (only if the image is color)
    cl::Buffer buf_lut_r, buf_lut_g, buf_lut_b;

    if (is_colour) {
        // Create LUTs for color channels (Red, Green, Blue)
        vector<unsigned char> lut_r(num_bins, 0), lut_g(num_bins, 0), lut_b(num_bins, 0);

        // Populate LUTs (for now just pass-through LUTs)
        for (int i = 0; i < num_bins; ++i) {
            lut_r[i] = static_cast<unsigned char>(i);
            lut_g[i] = static_cast<unsigned char>(i);
            lut_b[i] = static_cast<unsigned char>(i);
        }

        // Create OpenCL buffers for the LUTs
        buf_lut_r = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * num_bins, lut_r.data());
        buf_lut_g = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * num_bins, lut_g.data());
        buf_lut_b = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * num_bins, lut_b.data());
    }

    // Apply LUT to image using equalisation values (Grayscale or Color)
    cl::Kernel apply(program, is_colour ? "apply_lut_kernel_colour" : "apply_lut_kernel");

    apply.setArg(0, buf_input);
    apply.setArg(1, buf_output);

    if (is_colour) {
        apply.setArg(2, buf_lut_r);  // Red channel LUT
        apply.setArg(3, buf_lut_g);  // Green channel LUT
        apply.setArg(4, buf_lut_b);  // Blue channel LUT
    } else {
        apply.setArg(2, buf_lut);  // Grayscale LUT
    }

    if (is_colour){
    	apply.setArg(5, size);
    } else {
    	apply.setArg(3, size);
    }
    queue.enqueueNDRangeKernel(apply, cl::NullRange, cl::NDRange(size), cl::NullRange, nullptr, &evt_apply);

    // Read back equalised image
    cl::Event evt_read;
    int output_size = size * (is_colour ? 3 : 1);  // Adjusted for color images
    vector<unsigned char> equalised_image(output_size);  // Corrected size for the output buffer
    try {
        queue.enqueueReadBuffer(buf_output, CL_FALSE, 0, sizeof(unsigned char) * output_size, equalised_image.data(), nullptr, &evt_read);
        queue.finish();  // Wait for all GPU tasks to complete
    } catch (const cl::Error& e) {
        cout << "OpenCL Error during read buffer: " << e.what() << " (" << e.err() << ")" << endl;
        return -1;  // Exit in case of error
    }

    // Save the processed image to a file (output_equalised.ppm or output_equalised.pgm)
    CImg<unsigned char> result_image(equalised_image.data(), width, height, 1, (is_colour ? 3 : 1));
    result_image.save("output_equalised.ppm");  // You can specify a full path if you want to save it elsewhere

    // Optionally display the image after saving it
    result_image.display("Equalised Image");  // This will open the image in a viewer window

    // Output kernel timings using cl::Event
    cout << "\n=== Execution Times ===" << endl;
    cout << "Histogram kernel:       " << getEventDurationMS(evt_hist) << " ms" << endl;
    cout << "Scan kernel:            " << getEventDurationMS(evt_scan) << " ms" << endl;
    cout << "Normalise kernel:       " << getEventDurationMS(evt_norm) << " ms" << endl;
    cout << "Apply LUT kernel:       " << getEventDurationMS(evt_apply) << " ms" << endl;
    cout << "Read buffer (output):   " << getEventDurationMS(evt_read) << " ms" << endl;

    clock_t end_total = clock();  // End timing
    double total_time = getEventDurationMS(evt_hist);  // Total execution time using cl::Event

    // Output total execution time using printf
    cout << "\nHistogram Equalisation complete. Output saved to output_equalised." << (is_colour ? "ppm" : "pgm") << endl;
    cout << "Total Program Time:      " << total_time << " ms" << endl;

    return 0;
}

