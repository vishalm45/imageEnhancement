#include "CImg.h"
#include <CL/opencl.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <limits>
#include <cmath>

using namespace cimg_library;
using namespace std;

// helper function to get OpenCL event duration in milliseconds
double getEventDurationMS(const cl::Event& event) {
    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    return (end - start) * 1e-6;
}

pair<int, int> find_min_max_bin(const vector<cl_uint>& hist) {
    int min_bin = 0, max_bin = hist.size() - 1;
    while (min_bin < hist.size() && hist[min_bin] == 0) ++min_bin;
    while (max_bin >= 0 && hist[max_bin] == 0) --max_bin;
    return { min_bin, max_bin };
}

int main(int argc, char* argv[]) {
    auto start_total = chrono::high_resolution_clock::now();

    // Load image
    CImg<unsigned char> image("mdr16.ppm");
    int width = image.width(), height = image.height(), spectrum = image.spectrum();
    bool is_color = (spectrum == 3);

    const int num_bins = 256;
    int size = width * height;
    vector<unsigned char> output(size * spectrum);

    // Convert RGB to intensity (luminance): Y = 0.299R + 0.587G + 0.114B
    vector<unsigned char> luminance(size);
    cimg_forXY(image, x, y) {
        unsigned char R = image(x, y, 0, 0);
        unsigned char G = image(x, y, 0, 1);
        unsigned char B = image(x, y, 0, 2);
        luminance[y * width + x] = static_cast<unsigned char>(0.299 * R + 0.587 * G + 0.114 * B);
    }

    // OpenCL setup
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

    cl::Buffer buf_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * size, luminance.data());
    cl::Buffer buf_output(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * size);
    cl::Buffer buf_hist(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * num_bins);
    cl::Buffer buf_scan(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * num_bins);
    cl::Buffer buf_lut(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * num_bins);

    vector<cl_uint> hist_init(num_bins, 0);
    queue.enqueueWriteBuffer(buf_hist, CL_TRUE, 0, sizeof(cl_uint) * num_bins, hist_init.data());

    cl::Kernel histogram(program, "histogram_kernel");
    histogram.setArg(0, buf_input);
    histogram.setArg(1, buf_hist);
    histogram.setArg(2, size);
    histogram.setArg(3, num_bins);
    queue.enqueueNDRangeKernel(histogram, cl::NullRange, cl::NDRange(size));

    cl::Kernel scan(program, "scan_kernel");
    scan.setArg(0, buf_hist);
    scan.setArg(1, buf_scan);
    scan.setArg(2, num_bins);
    queue.enqueueNDRangeKernel(scan, cl::NullRange, cl::NDRange(num_bins));

    vector<cl_uint> scan_result(num_bins);
    queue.enqueueReadBuffer(buf_scan, CL_TRUE, 0, sizeof(cl_uint) * num_bins, scan_result.data());
    pair<int, int> minmax = find_min_max_bin(scan_result);
    int min_bin = minmax.first;
    int max_bin = minmax.second;

    cl::Kernel normalise(program, "normalise_kernel");
    normalise.setArg(0, buf_scan);
    normalise.setArg(1, buf_lut);
    normalise.setArg(2, num_bins);
    normalise.setArg(3, size);
    normalise.setArg(4, cl_int(min_bin));
    normalise.setArg(5, cl_int(max_bin));
    queue.enqueueNDRangeKernel(normalise, cl::NullRange, cl::NDRange(num_bins));

    cl::Kernel apply(program, "apply_lut_kernel");
    apply.setArg(0, buf_input);
    apply.setArg(1, buf_output);
    apply.setArg(2, buf_lut);
    apply.setArg(3, size);
    apply.setArg(4, num_bins);
    queue.enqueueNDRangeKernel(apply, cl::NullRange, cl::NDRange(size));

    vector<unsigned char> equalised_luminance(size);
    queue.enqueueReadBuffer(buf_output, CL_TRUE, 0, sizeof(unsigned char) * size, equalised_luminance.data());

    // Improved RGB rescaling based on luminance ratio
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
    result_image.save("output_equalised.ppm");

    auto end_total = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> total_time = end_total - start_total;

    cout << fixed << setprecision(3);
    cout << "\nPerceptual Histogram Equalisation complete. Output saved to output_equalised.ppm" << endl;
    cout << "Total Program Time:      " << total_time.count() << " ms" << endl;

    return 0;
}
