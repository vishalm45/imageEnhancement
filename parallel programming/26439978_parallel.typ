#set page(header: context [
  #set align(left)
  CMP3752 PARALLEL PROGRAMMING #h(1fr) Vishal Maisuria - 26439978
],
numbering: "1")
#set text(
  size: 14pt
)
= Introduction
To create a digital enhancement program, contrast adjustment using histogram eqalisation on a cumulative intensity histogram was implemented, which back-projects the original image's intensities to create an image of equalised intensities. The following report details how the program works and showcases the enhanced images, as well as the results and run times of the optimisations made.

= Implementation

#pagebreak()
= Introduction
This assignment implements a parallelised histogram equalisation algorithm, using OpenCL and C++, which enhaces contast for greyscale and an attempt at colour images. This program processes 8-bit images using GPU acceleration and evaluates performance via kernel execution times. 

Using classic parallel patterns such as histogram computation, inclusive scan and LUT mapping, I have tested the program using `.pgm` and `.ppm` images, with fixed and variable dimensions.

Histogram equalisation is a technique that adjust the pixel values of an image based on its intensity histogram @noauthor_histogram_nodate. This results in an enhanced image visibility by utilising the full dynamic range. 

= Implementation
To write the code, I used C++ and Visual Studio, along with the CImg library to load and save the greyscale (`.pgm`) and colour (`.ppm`) images. To perform the image processing steps in parallel, I used OpenCL, which allows for GPU acceleration. The code consists of platform and device selection (@fig:platformDeviceSelection), where platform selection finds the OpenCL platform, for example Intel or NVIDIA, and device selection finds the compute device, such as the CPU or GPU. Buffers (@fig:bufferCreation) are dynamically allocated according to the defined number of histogram bins (`num_bins`), enabling flexible control of histogram resolution. 

The OpenCL kernel code defines five main kernels. `histogram_kernel` computes the image histogram using atomic operations to avoid race conditions. `scan_kernel` performs an inclusive scan using the Hillis-Steele pattern to compute the cumulative histogram. `normalise_kernel` then normalises the cumulative histogram by scaling values using the detected minimum and maximum active bins to create a valid LUT (Lookup tables). Lookup tables are a predetermined array of numbers that provide a shortcut for a specific computation. `apply_lut_kernel` uses this LUT to remap the pixel values of the image to produce the equalised version. For colour images, `apply_lut_kernel_colour` handles the three channels (R, G, B) separately.

Beyond the code used from the workshops, I have also implemented a parameter called `num_bins`, where the histogram resolution is configurable, and the kernels are built to support this dynamically. Full colour support was added, including the RGB channel separation and combined LUT application. However, I was unable to complete this, as the colour images remained noisy and distorted. 





#figure(
  image("Screenshot 2025-04-04 at 18.28.09.png", width: 90%),
  caption: "Platform and Device Selection"
)<fig:platformDeviceSelection>

#figure(
  image("Screenshot 2025-04-04 at 19.08.56.png", width: 100%),
  caption: "Buffer creation"
)<fig:bufferCreation>

#figure(
  image("Screenshot 2025-04-04 at 20.12.07.png", width: 100%),
  caption: "Configuarable bins"
)<fig:configuarableBins>

#pagebreak()
#bibliography("bibliography.bib", style: "university-of-lincoln-harvard.csl", title: [References], full: true)
