#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>


/*
* Kernel for blurring an image
*/
__global__ 
void blurFilter(const unsigned char* inputimage, unsigned char* outputimage,
	int windowhalfsize, int imagemaxdimension)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// make sure we are in image boundary
	if (col < imagemaxdimension && row < imagemaxdimension)
	{
		int pixels = 0;
		float window_sum = 0;
		// loop over the window
		for (int i = -windowhalfsize; i <= windowhalfsize; i++)
		{
			for (int j = -windowhalfsize; j <= windowhalfsize; j++)
			{

				int current_col = col + i;
				int current_row = row + j;
				// make sure we are still in the image boundary
				if (current_col >= 0 && current_col < imagemaxdimension && current_row >= 0 && current_row < imagemaxdimension)
				{
					window_sum += inputimage[current_col + current_row * imagemaxdimension];
					pixels++;
				}
			}
		}
		// update pixel with the average
		outputimage[col + row * imagemaxdimension] = static_cast<unsigned char>(window_sum / pixels);
	}
}

/*
Catches errors returned from CUDA functions
*/
__host__
void errCatch(cudaError_t err) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char** argv)
{
	// Read cat image
	cv::Mat image = cv::imread("cat.jpg", cv::IMREAD_GRAYSCALE);

	if (!image.data || !image.isContinuous())                              // Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl;
		cv::waitKey(5000);
		return -1;
	}

	// show image
	cv::namedWindow("image", cv::WindowFlags::WINDOW_KEEPRATIO);
	cv::imshow("image", image);

	// blur window half width
	int window_half_size = 7;

	// Allocate memory for input and output data on the device
	const unsigned int width = image.cols;
	const unsigned int height = image.rows;
	const size_t input_size = width * height * sizeof(unsigned char);
	const size_t output_size = input_size;
	unsigned char* d_input = nullptr;
	unsigned char* d_output = nullptr;

	errCatch(cudaMalloc(&d_input, input_size));
	errCatch(cudaMalloc(&d_output, output_size));
	errCatch(cudaMemcpy(d_input, image.data, input_size, cudaMemcpyHostToDevice));

	// Blur image
	const dim3 blocksPerGrid(64, 64); // 4096 blocks
	const dim3 threadsPerBlock(16, 16); // 256 threads

	printf("%d blocks/grid\n", blocksPerGrid.x * blocksPerGrid.y);
	printf("%d threads/block\n", threadsPerBlock.x * threadsPerBlock.y);
	printf("%d total threads\n", blocksPerGrid.x * blocksPerGrid.y * threadsPerBlock.x * threadsPerBlock.y);
	
	// measure time
	auto const start = std::chrono::steady_clock::now();

	// Launch the CUDA kernel
	blurFilter<<< blocksPerGrid, threadsPerBlock>>>(d_input, d_output, window_half_size, width);
	errCatch(cudaDeviceSynchronize());

	// measure time
	auto const stop = std::chrono::steady_clock::now();
	auto const duration = stop - start;
	auto const duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
	auto const duration_us = std::chrono::duration_cast<std::chrono::microseconds>(duration);

	std::cout << "duration: " << duration_ms.count() << "." << duration_us.count() <<" ms" << std::endl;

	 
	// Copy output data from device memory to host memory
	unsigned char* output_data = new unsigned char[output_size];
	errCatch(cudaMemcpy(output_data, d_output, output_size, cudaMemcpyDeviceToHost));

	// Create an output image from the processed data and save it to disk
	cv::Mat output_image(height, width, CV_8UC1, output_data);
	cv::imwrite("output.jpg", output_image);

	// Display the output image
	cv::namedWindow("output", cv::WindowFlags::WINDOW_KEEPRATIO);
	cv::imshow("output", output_image);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Free device memory and host data
	errCatch(cudaFree(d_input));
	errCatch(cudaFree(d_output));
	delete[] output_data;

	// release images from memory
	output_image.release();
	image.release();
	return 0;
}