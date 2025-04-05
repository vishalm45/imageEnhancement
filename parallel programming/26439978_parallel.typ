#set page(header: context [
  #set align(left)
  CMP3752 PARALLEL PROGRAMMING #h(1fr) Vishal Maisuria - 26439978
],
numbering: "1")
#set text(
  size: 11pt
)
= Introduction
To create a digital enhancement program, contrast adjustment using histogram eqalisation on a cumulative intensity histogram was implemented, which back-projects the original image's intensities to create an image of equalised intensities. The following report details how the program works and showcases the enhanced images, as well as the results and run times of the optimisations made.

= Implementation
All processing is performend using OpenCL kernels on the GPU, and my program is executed via a `main` function that accepts command-line arguments for platform selection (`-p 0` or `-p 1`) to target the GPU or CPU respectively. The input files, which use `.pgm` or `.ppm` images, can directly be changed in the code, and this is what I have used to test when comparing both greyscale and colour images. These images are converted to 8-bit where necessary, and exception handling is implemented to catch both OpenCL and image loading errors. 

OpenCL begins by selecting a platform and device, creating a context and building the program from the `my_kernels.cl` kernel file. Buffers are created dynamically (@fig:buffercreation) depending on image size and `num_bins`. This also includes input and output buffers, histograms, cumulative histograms and LUTs (lookup tables). 

Greyscale images are processed directly, which proved effective in the results. For colour images, more configuration was needed, as the image is first converted to intensity using `Y = 0.299R + 0.587G + 0.114B`. Histogram equalisation is then applied to the Y channel only, and the resulting intensity is used to scale the original RGB values proportionally. This is meant to prevent colour distortion and allow perceptual enhancement, but I was unable to complete this, and the colour images remained noisy and unclear.

The `.cl` file contained my kernels, which included the `histogram_kernel`, which builds the histogram using atomic operaitons to avoid race conditions, and `scan_kernel`, which uses Hillis-Steele for cumulative histogram computation. `normalise_kernel` is the kernel that maps cumulative frequencies to the 0-255 range using `min_bin` and `max_bin`, which ensures the dynamic contrast enhancement. To apply the lookup tables, `apply_lut_kernel` and `apply_lut_kernel_colour` are used respectively. Each pixel in the original image serves as an index into the LUT, and the corresponding value is written into the output buffer. This mapping step effectively reassigns intensities based on the cumulative histogram, completing the equalisation process for the images.

Alongside support for colour images, I have imeplemented a number of features beyond the workshop exercises. The first of which is dynamic bin count, where the histogram size is configurable using the `num_bins` parameter. this is beneficial as it allows for flexibility and control over the contrast resolution depending on the image type and bit depth. 8-bit images typically use 256 bins (0-255), but 16-bit images may require up to 65,536 bins, and a dynamic bin count allows for the algorithm to scale accordingly without hardcoding for each case. Additionally, I have implemented minimum bin and maximum bin support, where the LUT is scaled based on the actual used intensity range. To measure the kernel and total time, I used OpenCL's `cl::Event` and `chrono` from C++. This allows for critical analysis, and is vital for optimising parallel operations.

= Results

#pagebreak()
#grid(
  columns: 2,
  rows: 4,
  gutter: 5pt,
  
  figure(
    image("image-4.png", width: 70%),
    caption: "Original greyscale image"
  ),

  figure(
    image("image-2.png", width: 70%),
    caption: "Greyscale image after enhancement"
  ),

  figure(
    image("image-5.png", width: 50%, height: 30%),
    caption: "Original colour image"
  ),

  figure(
    image("image.png",width: 50%, height: 30%),
    caption: "Colour image after enhancement"
  ),
)

#figure(
  image("image-3.png", width: 120%),
  caption: "Times taken in ms"
)

#figure(
  image("Screenshot 2025-04-04 at 19.08.56.png", width: 100%),
  caption: "Buffer creation"
)<fig:buffercreation>

#figure(
  image("Screenshot 2025-04-04 at 20.12.07.png", width: 100%),
  caption: "Configuarable number of bins"
)<fig:configbins>

#figure(
  image("image-6.png"),
  caption: "Hillis-Steele algorithm used on the cumulative histogram"  
)

#figure(
  image("image-7.png"),
  caption: "Platform and device selection"
)
#pagebreak()

For visualisation purposes, I created a histogram and cumulative histogram graph from the data by converting it to a CSV file and then using Python and Matplotlib. In the histogram graph, it shows the frequency of pixel intensity values in the image. The tall, narrow spike at around 135 shows that the vast majority of pixels have mid-range brightness. The cumulative histogram graph shows the cumulative sum of pixel intensities across the image. It stays flat until around 120, which shows that there are very few dark pixels, and then there is a steep slope between \~130 and 170, which shows that there is a rapid accumulation of pixels, where most of the image information lies. 
#image("histogram.png")
#image("cumulative_histogram.png")
