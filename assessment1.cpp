#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>

#include "Utils.h"
#include "CImg.h"
#include <CL/opencl.hpp>

using namespace std;
using namespace cimg_library;

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
    int platform_index = -1; // Initialize to -1 to indicate no preference
    if (argc == 3 && string(argv[1]) == "-p") {
        platform_index = atoi(argv[2]);
    }

    CImg<unsigned char> image("mdr16.ppm");
    int width = image.width(), height = image.height(), spectrum = image.spectrum();
    bool is_colour = (spectrum == 3);
    const int num_bins = 256;
    int size = width * height;
    vector<unsigned char> output(size * (is_colour ? 3 : 1));
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
            luminance[y * width + x] = image(x, y);
        }
    }

    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        cerr << "No OpenCL platforms found." << endl;
        return -1;
    }

    cout << "Available OpenCL Platforms:" << endl;
    for (size_t i = 0; i < platforms.size(); ++i) {
        cout << "  Platform " << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;
        vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (!devices.empty()) {
            cout << "    Devices:" << endl;
            for (size_t j = 0; j < devices.size(); ++j) {
                cout << "      Device " << j << ": " << devices[j].getInfo<CL_DEVICE_NAME>() << " (Type: ";
                cl_device_type type = devices[j].getInfo<CL_DEVICE_TYPE>();
                if (type & CL_DEVICE_TYPE_CPU) cout << "CPU";
                if (type & CL_DEVICE_TYPE_GPU) cout << "GPU";
                if (type & CL_DEVICE_TYPE_ACCELERATOR) cout << "ACCELERATOR";
                if (type & CL_DEVICE_TYPE_DEFAULT) cout << "DEFAULT";
                cout << ")" << endl;
            }
        } else {
            cout << "    No devices found for this platform." << endl;
        }
    }

    cl::Platform selected_platform;
    if (platform_index >= 0 && platform_index < platforms.size()) {
        selected_platform = platforms[platform_index];
        cout << "\nUsing Platform " << platform_index << ": " << selected_platform.getInfo<CL_PLATFORM_NAME>() << endl;
    } else {
        selected_platform = platforms.front();
        cout << "\nUsing default Platform 0: " << selected_platform.getInfo<CL_PLATFORM_NAME>() << endl;
    }

    vector<cl::Device> devices;
    cl_device_type device_type = CL_DEVICE_TYPE_GPU;
    if (argc == 3 && string(argv[1]) == "-p" && atoi(argv[2]) == 1) {
        device_type = CL_DEVICE_TYPE_CPU;
        cout << "Attempting to use CPU device." << endl;
    } else {
        cout << "Attempting to use GPU device (default)." << endl;
    }

    try {
        selected_platform.getDevices(device_type, &devices);
    } catch (const cl::Error& err) {
        cerr << "Error getting devices of type " << (device_type == CL_DEVICE_TYPE_CPU ? "CPU" : "GPU")
             << ": " << err.what() << " (" << err.err() << ")" << endl;
        return -1;
    }

    if (devices.empty()) {
        cerr << "No OpenCL devices of the requested type found." << endl;
        return -1;
    }
    cl::Device device = devices.front();
    cout << "Using Device: " << device.getInfo<CL_DEVICE_NAME>() << endl;

    cl::Context context({ device });
    cl::Program::Sources sources;
    ifstream kernel_file("my_kernels.cl");
    string kernel_code((istreambuf_iterator<char>(kernel_file)), istreambuf_iterator<char>());
    sources.push_back({ kernel_code.c_str(), kernel_code.length() });

    cl::Program program(context, sources);
    try {
        program.build({ device });
    } catch (const cl::Error& err) {
        cerr << "Error building program: " << err.what() << " (" << err.err() << ")" << endl;
        cerr << "Build log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        return -1;
    }

    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    cl::Buffer buf_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * size, luminance.data());
    cl::Buffer buf_output(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * size * (is_colour ? 3 : 1));
    cl::Buffer buf_hist(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * num_bins);
    cl::Buffer buf_scan(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * num_bins);
    cl::Buffer buf_lut(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * num_bins);

    try {
        queue.enqueueFillBuffer(buf_hist, 0, 0, sizeof(cl_uint) * num_bins);
        queue.finish();
        cout << "buf_hist filled with zeros." << endl;
    } catch (const cl::Error& err) {
        cerr << "Error filling buf_hist: " << err.what() << " (" << err.err() << ")" << endl;
        return -1;
    }

    cl::Event evt_hist, evt_scan, evt_norm, evt_apply, evt_read;
    cl::Kernel histogram(program, "histogram_kernel");
    histogram.setArg(0, buf_input);
    histogram.setArg(1, buf_hist);
    histogram.setArg(2, size);
    histogram.setArg(3, num_bins);
    try {
        queue.enqueueNDRangeKernel(histogram, cl::NullRange, cl::NDRange(size), cl::NullRange, nullptr, &evt_hist);
        queue.finish();
        cout << "Histogram kernel executed." << endl;
    } catch (const cl::Error& err) {
        cerr << "Error running histogram kernel: " << err.what() << " (" << err.err() << ")" << endl;
        return -1;
    }

    cl::Kernel scan(program, "scan_kernel");
    scan.setArg(0, buf_hist);
    scan.setArg(1, buf_scan);
    scan.setArg(2, num_bins);
    try {
        queue.enqueueNDRangeKernel(scan, cl::NullRange, cl::NDRange(num_bins), cl::NullRange, nullptr, &evt_scan);
        queue.finish();
        cout << "Scan kernel executed." << endl;
    } catch (const cl::Error& err) {
        cerr << "Error running scan kernel: " << err.what() << " (" << err.err() << ")" << endl;
        return -1;
    }

    vector<cl_uint> scan_result(num_bins);
    try {
        queue.enqueueReadBuffer(buf_scan, CL_TRUE, 0, sizeof(cl_uint) * num_bins, scan_result.data());
        cout << "Read from buf_scan successful." << endl;
        cout << "First few scan_result: ";
        for (int i = 0; i < min(10, (int)scan_result.size()); ++i) {
            cout << scan_result[i] << " ";
        }
        cout << endl;
    } catch (const cl::Error& err) {
        cerr << "Error reading buf_scan: " << err.what() << " (" << err.err() << ")" << endl;
        return -1;
    }

    pair<int, int> minmax = find_min_max_bin(scan_result);
    int min_bin = minmax.first;
    int max_bin = minmax.second;
    cout << "Min Bin: " << min_bin << ", Max Bin: " << max_bin << endl;

    cl::Kernel normalise(program, "normalise_kernel");
    normalise.setArg(0, buf_scan);
    normalise.setArg(1, buf_lut);
    normalise.setArg(2, num_bins);
    normalise.setArg(3, size);
    normalise.setArg(4, cl_int(min_bin));
    normalise.setArg(5, cl_int(max_bin));
    try {
        queue.enqueueNDRangeKernel(normalise, cl::NullRange, cl::NDRange(num_bins), cl::NullRange, nullptr, &evt_norm);
        queue.finish();
        cout << "Normalise kernel executed." << endl;
    } catch (const cl::Error& err) {
        cerr << "Error running normalise kernel: " << err.what() << " (" << err.err() << ")" << endl;
        return -1;
    }

    cl::Buffer buf_lut_r, buf_lut_g, buf_lut_b;

    if (is_colour) {
        vector<unsigned char> lut_r(num_bins, 0), lut_g(num_bins, 0), lut_b(num_bins, 0);
        for (int i = 0; i < num_bins; ++i) {
            lut_r[i] = static_cast<unsigned char>(i);
            lut_g[i] = static_cast<unsigned char>(i);
            lut_b[i] = static_cast<unsigned char>(i);
        }
        buf_lut_r = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * num_bins, lut_r.data());
        buf_lut_g = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * num_bins, lut_g.data());
        buf_lut_b = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * num_bins, lut_b.data());
    }

    cl::Kernel apply(program, is_colour ? "apply_lut_kernel_colour" : "apply_lut_kernel");
    apply.setArg(0, buf_input);
    apply.setArg(1, buf_output);
    if (is_colour) {
        apply.setArg(2, buf_lut_r);
        apply.setArg(3, buf_lut_g);
        apply.setArg(4, buf_lut_b);
        apply.setArg(5, size);
    } else {
        apply.setArg(2, buf_lut);
        apply.setArg(3, size);
    }
    try {
        queue.enqueueNDRangeKernel(apply, cl::NullRange, cl::NDRange(size), cl::NullRange, nullptr, &evt_apply);
        queue.finish();
        cout << "Apply kernel executed." << endl;
    } catch (const cl::Error& err) {
        cerr << "Error running apply kernel: " << err.what() << " (" << err.err() << ")" << endl;
        return -1;
    }

    int output_size = size * (is_colour ? 3 : 1);
    vector<unsigned char> equalised_image(output_size);
    try {
        queue.enqueueReadBuffer(buf_output, CL_FALSE, 0, sizeof(unsigned char) * output_size, equalised_image.data(), nullptr, &evt_read);
        queue.finish();
        cout << "Read from buf_output successful." << endl;
    } catch (const cl::Error& err) {
        cerr << "Error reading buf_output: " << err.what() << " (" << err.err() << ")" << endl;
        return -1;
    }

    CImg<unsigned char> result_image(equalised_image.data(), width, height, 1, (is_colour ? 3 : 1));
    result_image.save("output_equalised.ppm");
    result_image.display("Equalised Image");

    cout << "\n=== Execution Times ===" << endl;
    cout << "Histogram kernel:          " << getEventDurationMS(evt_hist) << " ms" << endl;
    cout << "Scan kernel:             " << getEventDurationMS(evt_scan) << " ms" << endl;
    cout << "Normalise kernel:          " << getEventDurationMS(evt_norm) << " ms" << endl;
    cout << "Apply LUT kernel:          " << getEventDurationMS(evt_apply) << " ms" << endl;
    cout << "Read buffer (output):      " << getEventDurationMS(evt_read) << " ms" << endl;

    double total_time = getEventDurationMS(evt_hist) + getEventDurationMS(evt_scan) + getEventDurationMS(evt_norm) + getEventDurationMS(evt_apply) + getEventDurationMS(evt_read);
    cout << "\nHistogram Equalisation complete. Output saved to output_equalised." << (is_colour ? "ppm" : "pgm") << endl;
    cout << "Total Program Time:          " << total_time << " ms" << endl;

    return 0;
}
