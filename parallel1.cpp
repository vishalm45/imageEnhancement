// ========== Tutorial 3.cpp (Histogram Equalisation Host Code with Timing) ==========

#include "CImg.h"
#include <CL/opencl.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

using namespace cimg_library;
using namespace std;

// Helper function to get OpenCL event duration in milliseconds
double getEventDurationMS(const cl::Event& event) {
    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    return (end - start) * 1e-6; // nanoseconds to milliseconds
}

int main() {
    auto start_total = chrono::high_resolution_clock::now();

    // Load grayscale image (PGM format)
    CImg<unsigned char> image("test_large.pgm"); // Replace with "test.pgm" for smaller image
    int width = image.width(), height = image.height();
    int size = width * height;
    vector<unsigned char> input(image.data(), image.data() + size);
    vector<unsigned char> output(size);

    // Setup OpenCL
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();

    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    cl::Context context({ device });
    cl::Program::Sources sources;

    ifstream kernel_file("my_kernels.cl");
    string kernel_code((istreambuf_iterator<char>(kernel_file)), istreambuf_iterator<char>());
    sources.push_back({ kernel_code.c_str(), kernel_code.length() });

    cl::Program program(context, sources);
    program.build({ device });
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Create OpenCL buffers (ensure correct byte sizes)
    cl::Buffer buf_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * size, input.data());
    cl::Buffer buf_output(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * size);
    cl::Buffer buf_hist(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * 256);
    cl::Buffer buf_scan(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * 256);
    cl::Buffer buf_lut(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * 256);

    // Zero-initialize histogram
    vector<cl_uint> hist_init(256, 0);
    cl::Event evt_write_hist;
    queue.enqueueWriteBuffer(buf_hist, CL_TRUE, 0, sizeof(cl_uint) * 256, hist_init.data(), nullptr, &evt_write_hist);

    // Kernel 1: Histogram
    cl::Event evt_hist;
    cl::Kernel histogram(program, "histogram_kernel");
    histogram.setArg(0, buf_input);
    histogram.setArg(1, buf_hist);
    histogram.setArg(2, size);
    queue.enqueueNDRangeKernel(histogram, cl::NullRange, cl::NDRange(size), cl::NullRange, nullptr, &evt_hist);

    // Kernel 2: Scan
    cl::Event evt_scan;
    cl::Kernel scan(program, "scan_kernel");
    scan.setArg(0, buf_hist);
    scan.setArg(1, buf_scan);
    scan.setArg(2, 256);
    queue.enqueueNDRangeKernel(scan, cl::NullRange, cl::NDRange(256), cl::NullRange, nullptr, &evt_scan);

    // Kernel 3: Normalize cumulative histogram
    cl::Event evt_norm;
    cl::Kernel normalize(program, "normalize_kernel");
    normalize.setArg(0, buf_scan);
    normalize.setArg(1, buf_lut);
    normalize.setArg(2, 256);
    normalize.setArg(3, size);
    queue.enqueueNDRangeKernel(normalize, cl::NullRange, cl::NDRange(256), cl::NullRange, nullptr, &evt_norm);

    // Kernel 4: Apply LUT to input image
    cl::Event evt_apply;
    cl::Kernel apply(program, "apply_lut_kernel");
    apply.setArg(0, buf_input);
    apply.setArg(1, buf_output);
    apply.setArg(2, buf_lut);
    apply.setArg(3, size);
    queue.enqueueNDRangeKernel(apply, cl::NullRange, cl::NDRange(size), cl::NullRange, nullptr, &evt_apply);

    // Read result
    cl::Event evt_read;
    queue.enqueueReadBuffer(buf_output, CL_TRUE, 0, sizeof(unsigned char) * size, output.data(), nullptr, &evt_read);

    // Save result image
    CImg<unsigned char> result_image(output.data(), width, height, 1, 1);
    result_image.save("output_equalised.pgm");

    // Total execution time
    auto end_total = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> total_time = end_total - start_total;

    // Output performance data
    cout << fixed << setprecision(3);
    cout << "\n--- Execution Times ---" << endl;
    cout << "Memory Transfer (init):  " << getEventDurationMS(evt_write_hist) << " ms" << endl;
    cout << "Histogram kernel:        " << getEventDurationMS(evt_hist) << " ms" << endl;
    cout << "Scan kernel:             " << getEventDurationMS(evt_scan) << " ms" << endl;
    cout << "Normalize kernel:        " << getEventDurationMS(evt_norm) << " ms" << endl;
    cout << "Apply LUT kernel:        " << getEventDurationMS(evt_apply) << " ms" << endl;
    cout << "Read buffer (output):    " << getEventDurationMS(evt_read) << " ms" << endl;
    cout << "Total Program Time:      " << total_time.count() << " ms" << endl;

    cout << "\nHistogram Equalisation complete. Output saved to output_equalised.pgm" << endl;
    return 0;
}
