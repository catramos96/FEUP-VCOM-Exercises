#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<iostream>
#include<conio.h>					// may have to modify this line if not using Windows
#include <stdio.h>
#include "Auxiliary.h"

using namespace std;
using namespace cv;

int main() {
	
													/****************
													*		1		*
													****************/

	//NOTE: Mat stores the image as BGR order

	//=====================================================================================================================
	// #### 1.A ####

	// Read the name of a file containing an image in 'jpg' format and show it in a window, whose name is the name of the
	// file.Test whether the image was successfully read.Display the height and width of the image, on the console

	/*
	cv::Mat image = cv::imread("image.jpg");

	cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
	cv::imshow("image", image);

	cout << "Width: " << image.size().width << endl << "Height: " << image.size().height << endl;
	*/

	//=====================================================================================================================
	// #### 1.B ####

	// Read a color image in 'jpg' format and save it in 'bmp' format.

	/*
	cv::Mat image = cv::imread("image.jpg");
	cv::imwrite("image2.bmp", image);
	*/

	//=====================================================================================================================
	// #### 1.C ####

	// Read an image from a file, allow the user to select a region of interest (ROI) in the image, by clicking on two points
	// that identify two opposite corners of the selected ROI, and save the ROI into another file.

	/*
	cv::Mat image = cv::imread("image.jpg");
	Rect2d r = selectROI(image);	//select ROI
	Mat imCrop = image(r);			//crop image
	cv::imshow("cropped", imCrop);
	cv::imwrite("cropped.bmp", image);
	*/

	//=====================================================================================================================

													/****************
													*		2		*
													****************/
	/*
	  (0,0)--------->(width,0)
	    |				 |
	    |				 |
		|	  IMAGE		 |
	    |				 |
	    v				 |
	(height,0)-----(height,width)
	*/

	//=====================================================================================================================
	// #### 2.A ####

	// Create a grayscale image, having 100(lines)x200(columns) pixels with constant intensity, 100; draw the two
	// diagonals of the image with intensity 255. Display the image.

	/*
	Mat grayscale_image(
		100,			//height
		200,			//width
		CV_8UC1,		//type		8U: 8bit Unsigned integer	C1: 1 channel
		Scalar(100));	//intensity of channels

	int width = grayscale_image.size().width;
	int height = grayscale_image.size().height;

	//Drawing the diagonals
	line(grayscale_image, Point(0,0), Point(width,height), Scalar(255, 255, 255), 1, 8, 0);
	line(grayscale_image, Point(width, 0), Point(0, height), Scalar(255, 255, 255), 1, 8, 0);

	imshow("grayscale", grayscale_image);
	*/

	//=====================================================================================================================
	// #### 2.B ####

	// Create a color image, having 50(lines)x200(columns) pixels with constant intensity, 100; draw the two diagonals of
	// the image, one in red color, the other in blue color.Display the image.

	/*
	Mat color_image(
		50,						//height
		200,					//width
		CV_8UC3,				//type		8U: 8bit Unsigned integer	C3: 3 channel (RGB)
		Scalar(100,100,100));	//intensity of channels

	int width = color_image.size().width;
	int height = color_image.size().height;

	//Drawing the diagonals
	line(color_image, Point(0, 0), Point(width, height), Scalar(255, 0, 0), 1, 8, 0);	//RED
	line(color_image, Point(width, 0), Point(0, height), Scalar(0, 0, 255), 1, 8, 0);	//BLUE

	imshow("color", color_image);
	*/

	//=====================================================================================================================
	// #### 2.C ####

	// Read a color image, display it in one window, convert it to grayscale, display the grayscale image in another window
	// and save the grayscale image to a different file.

	/*
	Mat color_image = cv::imread("colors.jpg");
	imshow("Color Image", color_image);

	Mat grayscale_image(color_image.size().width, color_image.size().height, CV_8UC1);
	cvtColor(color_image, grayscale_image, cv::COLOR_RGB2GRAY);

	imshow("Grayscale Image", grayscale_image);
	*/

	//=====================================================================================================================
	// #### 2.D ####

	// Read an image (color or grayscale) and add "salt and pepper" noise to it. The number of noisy points must be 10%
	// of the total number of image points.Suggestion: start by determining the number of image channels.

	/*
	Mat image = imread("colors.jpg");

	double noise_ratio = 0.1;
	cout << "Noise ratio: " << noise_ratio << endl;

	imshow("Before Noise", image);

	SaltAndPepper(&image, noise_ratio);
		
	imshow("After Noise", image);
	*/

	//=====================================================================================================================
	// #### 2.E ####

	// Read a color image (in RGB format), split the 3 channels and show each channel in a separate window. Add a
	// constant value to one of the channels, merge the channels into a new color image and show the resulting image.

	/*
	Mat image = imread("colors.jpg",CV_LOAD_IMAGE_COLOR);

	vector<Mat> channels_image;

	split(image, channels_image);

	imshow("Red Channel", channels_image[0]);
	imshow("Green Channel", channels_image[1]);
	imshow("Blue Channel", channels_image[2]);

	// Set to constant value
	channels_image[0].setTo(Scalar(100));
	imshow("Red Channel with constant value", channels_image[0]);

	Mat merged_image;

	merge(channels_image, merged_image);
	imshow("Merged Image", merged_image);
	*/
	
	//=====================================================================================================================
	// #### 2.F ####

	// Read a color image (in RGB format), convert it to HSV, split the 3 HSV channels and show each channel in a
	// separate window.Add a constant value to saturation channel, merge the channels into a new color image and show
	// the resulting image.

	/*
	Mat color_image = imread("colors.jpg", CV_LOAD_IMAGE_COLOR);

	//convert to hsv
	Mat hsv_image;
	cvtColor(color_image, hsv_image, cv::COLOR_RGB2HSV);

	//split the 3 channels
	vector<Mat> channels;
	split(hsv_image, channels);

	//show channels
	imshow("Hue Channel", channels[0]);
	imshow("Saturation Channel", channels[1]);
	imshow("Value Channel", channels[2]);

	//saturation with constant value
	channels[1].setTo(Scalar(100));

	//merge channels
	Mat merged_image;
	merge(channels, merged_image);
	imshow("Merged Image", merged_image);
	*/

	//=====================================================================================================================

													/****************
													*		3		*
													****************/

	//=====================================================================================================================
	// #### 3.A ####

	// Display a video acquired from the webcam (in color) in one window and acquire and save a frame when the user
	// presses the keyboard.Show the acquired frame in another window

	/*
	VideoCapture cap(0);	// open the default camera
	if (!cap.isOpened())	// check if we succeeded
		return -1;

	for (;;)
	{
		//display camera video
		Mat frame;
		cap >> frame;					
		imshow("Camera", frame);

		//keyboard input
		if (waitKey(50) >= 0) {
			imshow("Captured", frame);
			imwrite("captured_frame.jpg", frame);
		}
	}

	cap.release();	//closes file or camera
	*/

	//=====================================================================================================================
	// #### 3.B ####

	// Display the video acquired from the webcam (in color) in one window and the result of the conversion of each
	// frame to grayscale in another window.

	/*
	VideoCapture cap(0);	// open the default camera
	if (!cap.isOpened())	// check if we succeeded
	return -1;

	for (;;)
	{
		//display camera color
		Mat frame_color, frame_gray;
		cap >> frame_color;
		imshow("Color Camera", frame_color);

		//display camera gray
		cvtColor(frame_color, frame_gray, CV_RGB2GRAY);
		imshow("Gray Camera", frame_gray);

		waitKey(50);
	}

	cap.release();	//closes file or camera
	*/

	//=====================================================================================================================
	// #### 3.C ####

	// Modify the program developed in b) so that the resulting frames are in binary format (intensity of each pixel is 0 or
	// 255); use a threshold value of 128.

	/*
	VideoCapture cap(0);	// open the default camera
	if (!cap.isOpened())	// check if we succeeded
		return -1;

	for (;;)
	{
		//display camera color
		Mat frame_color, frame_binary;
		cap >> frame_color;
		imshow("Color Camera", frame_color);

		//display camera gray
		cvtColor(frame_color, frame_binary, CV_RGB2GRAY);
		frame_binary.setTo(0, frame_binary < 128);
		frame_binary.setTo(255, frame_binary >= 128);
		imshow("Binary Camera", frame_binary);

		waitKey(50);
	}

	cap.release();	//closes file or camera
	*/

	//=====================================================================================================================

													/****************
													*		4		*
													****************/

	//=====================================================================================================================
	// #### 4.A ####

	// Take a low contrast image and plot its histogram.

	/*
	Mat image = imread("low_contrast.jpg");
	imshow("LOW CONTRAST IMAGE", image);

	Mat histogram = GetHistogram(image);

	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histogram);
	*/		

	//=====================================================================================================================
	// #### 4.B.1 ####

	// Enhance the image constrast using simple histogram equalization and show the resulting enhanced images and their histograms.
	
	/*
	Mat image = imread("low_contrast.jpg",0);
	
	imshow("NORMAL IMAGE", image);
	imshow("NORMAL IMAGE HISTOGRAM", GetHistogram(image));

	ApplyHistogramEqualization(&image);

	imshow("EQUALIZED IMAGE", image);
	imshow("EQUALIZED IMAGE HISTOGRAM", GetHistogram(image));
	*/

	//=====================================================================================================================
	// #### 4.B.2 ####

	// Enhance the image constrast using CLAHE and show the resulting enhanced images and their histograms.

	/*
	Mat image = imread("low_contrast.jpg");
	
	imshow("ORIGINAL", image);
	Mat original_histogram = GetHistogram(image);
	imshow("ORIGINAL HISTOGRAM", original_histogram);

	ApplyClahe(&image, 2.0);

	imshow("CLAHE", image);
	Mat clahe_histogram = GetHistogram(image);
	imshow("CLAHE HISTOGRAM", clahe_histogram);
	*/

	//=====================================================================================================================

													/****************
													*		5		*
													****************/

	//=====================================================================================================================
	// #### 5 ####

	// Take a noisy image and filter it (try different filter sizes), using a mean filter, gaussian filter, median nfilter and bilateral.

	/*
	Mat image = imread("noisy.jpg");
	Mat filter_image = Mat();

	int MAX_KERNEL_LENGTH = 5;

	imshow("ORIGINAL", image);
	
	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
		blur(image, filter_image, Size(i, i), Point(-1, -1));

	imshow("MEDIUM FILTER", filter_image);

	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
		GaussianBlur(image, filter_image, Size(i, i), 0, 0);

	imshow("GAUSSIAN FILTER", filter_image);
	
	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
		medianBlur(image, filter_image, i);

	imshow("MEDIAN FILTER", filter_image);

	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
		bilateralFilter(image, filter_image, i, i * 2, i / 2);

	imshow("BILATERAL FILTER", filter_image);	
	*/

	//=====================================================================================================================

													/****************
													*		6		*
													****************/

	//=====================================================================================================================
	// #### 6.A ####

	// Detect the edges of an image using the Sobel filter (try different thresholds
	//http://opencvexamples.blogspot.com/2013/10/sobel-edge-detection.html

	/*
	Mat image = imread("hand.jpg",0);
	Mat edges = Mat();
	Mat edges_image_x = Mat();
	Mat edges_image_y = Mat();

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	//Reduce noise with kernel
	GaussianBlur(image, image, Size(3, 3), 0, 0, BORDER_DEFAULT);

	//Sobel in both directions
	Sobel(image, edges_image_x, ddepth, 1, 0, 3, scale, delta);
	convertScaleAbs(edges_image_x, edges_image_x);

	Sobel(image, edges_image_y, ddepth, 0, 1, 3, scale, delta);
	convertScaleAbs(edges_image_y, edges_image_y);
	
	//Approximate gradients
	addWeighted(edges_image_x, 0.5, edges_image_y, 0.5, 0, edges);

	normalize(edges, edges, 0, 255, NORM_MINMAX);
	threshold(edges,edges, 0, 255, THRESH_BINARY | THRESH_OTSU);
	
	imshow("ORIGINAL", image);
	imshow("EDGES", edges);
	*/

	//=====================================================================================================================
	// #### 6.B ####

	// Detect the edges of an image using the Canny filter (try different thresholds
	//https://docs.opencv.org/3.0-beta/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html

	/*
	Mat image = imread("hand.jpg",0);
	Mat edges_image = Mat();

	// Reduce noise with a kernel 3x3
	blur(image, image, Size(3, 3));

	// Canny detector
	int threshold = 100;
	int apperture = 3;
	Canny(image, edges_image, threshold, threshold*3, apperture);

	imshow("ORIGINAL", image);
	imshow("EDGES", edges_image);
	*/

	//=====================================================================================================================
	// #### 6.D ####	INACABADO

	// Detect the edges of an image using the Laplacian filter; try different apertures;
	// notes:1) in order to visualize the result it may be necessary to rescale the resulting values;
	// 2) to isolate the edges it is necessary to detect the zero crossings in the result

	/*
	Mat image = imread("cat.jpg",0);
	Mat final = Mat();

	int scale = 1;
	int delta = 0;
	int ddepth = CV_32F;
	int kernel = 3;	

	// Remove noise by blurring with a Gaussian filter
	GaussianBlur(image, image, Size(3, 3), 0, 0, BORDER_DEFAULT);

	// apply filter
	Laplacian(image, final, ddepth, kernel, scale, delta);
	//convertScaleAbs(final, final);

	//Find zero crossings
	Mat edges = Mat::zeros(final.rows, final.cols, CV_8UC1);

	/*
	for (int x = 0; x < final.cols; x++) {
		for (int y = 0; y < final.rows; y++) {

			int diff_signed = 0;

			//borders
			if (x + 1 >= final.rows || x - 1 < 0 || y + 1 >= final.cols || y - 1 < 0) 
				continue;
			else {

				if ((signed(final.at<int>(x, y))) != (signed(final.at<int>(x + 1, y))))
					diff_signed++;
				if ((signed(final.at<int>(x, y))) != signed(final.at<int>(x, y + 1)))
					diff_signed++;
				if ((signed(final.at<int>(x, y))) != signed(final.at<int>(x, y - 1)))
					diff_signed++;
				if ((signed(final.at<int>(x, y))) != signed(final.at<int>(x - 1, y)))
					diff_signed++;
			}

			if(diff_signed >=2)
				edges.at<uchar>(x, y) = 255;
			
			if (final.at<int>(x, y) == 0)
				edges.at<uchar>(x, y) = 255;
				
		}
	}
	*/
	/*
	imshow("ORIGINAL", image);
	imshow("LAPACIAN", final);
	//imshow("ZERO CROSSINGS", edges);
	*/

	//=====================================================================================================================

												/****************
												*		7		*
												****************/

	//=====================================================================================================================
	// #### 7.A ####

	// Compare the functionality of HoughLines() and HoughLinesP() OpenCV functions for line detection

	/*
	HoughLinesP:	Finds line segments in a binary image using the probabilistic Hough transform.
	HoughLines:		Finds lines in a binary image using the standard Hough transform.
	*/

	//=====================================================================================================================
	// #### 7.B ####

	// Use HoughLines() to detect lines in a binary image; try different parameter values; draw the detected lines on the
	// image, using line().

	/*
	Mat image = imread("hand.jpg", 0);
	Mat final = Mat(), final_lines = Mat();
	imshow("ORIGINAL", image);

	//detect edges
	GaussianBlur(image, image, Size(3, 3),0,0);
	int threshold = 100;
	int apperture = 3;

	Canny(image, final, threshold, threshold * 3, apperture);
	cvtColor(final, final_lines, CV_GRAY2BGR);
	imshow("CANNY", final);

	vector<Vec2f> lines;
	HoughLines(final, lines, 1, CV_PI / 180, 100, 0, 0);
	
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(final_lines, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
	}
	
	imshow("HOUGH LINES", final_lines);
	*/

	//=====================================================================================================================
	// #### 7.C ####

	// Use HoughLinesP() to detect line segments in a binary image; try different parameter values; draw the detected
	// line segments on the image.

	
	Mat image = imread("hand.jpg", 0);
	Mat final = Mat(), final_lines = Mat();
	imshow("ORIGINAL", image);

	//detect edges
	GaussianBlur(image, image, Size(3, 3), 0, 0);
	int threshold = 100;
	int apperture = 3;

	Canny(image, final, threshold, threshold * 3, apperture);
	cvtColor(final, final_lines, CV_GRAY2BGR);
	imshow("CANNY", final);

	vector<Vec4i> lines;
	HoughLinesP(final, lines, 1, CV_PI / 180, 80, 30, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		line(image, Point(lines[i][0], lines[i][1]),
			Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
	}

	imshow("HOUGH LINES P", image);
	

	//=====================================================================================================================
	// #### 7.D ####

	// Take an image containing coins and use HoughCircles() to detect the coins in the image
	
	/*
	Mat src, src_gray;

	/// Read the image
	src = imread("coins.jpg", 1);

	/// Convert it to gray
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Reduce the noise so we avoid false circle detection
	//GaussianBlur(src_gray, src_gray, Size(3, 3), 2, 2);

	vector<Vec3f> circles;

	/// Apply the Hough Transform to find the circles
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, 100, 0, 0);

	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
		cout << i << "/" << circles.size() << endl;
	}

	imshow("Hough Circle Transform Demo", src);
	*/
	cv::waitKey(0);                 // hold windows open until user presses a key

	return(0);
}

