#include "Auxiliary.h"

Mat GetHistogram(Mat image) {

	//split the channels
	vector<Mat> channels;
	split(image, channels);

	//from 0 to 256
	int hist_size = 256;
	float range[] = { 0,256 };	//range value per pixel
	const float *hist_range = { range };

	bool uniform = true;		//bins with the same size
	bool accumulate = false;	//clear histograms in the beggining

	vector<Mat> channels_histogram;

	// void calcHist(const Mat* images, int n_images, const int* channels, InputArray mask, OutputArray hist, 
	// int dims, const int* histSize, const float** ranges, bool uniform=true, bool accumulate=false )

	for (int i = 0; i < channels.size(); i++) {
		channels_histogram.push_back(Mat());
		calcHist(&channels[i], 1, 0, Mat(), channels_histogram[i], 1, &hist_size, &hist_range, uniform, accumulate);
	}

	// Draw the histograms for each channel
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / hist_size);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	// Normalize the result to [ 0, histImage.rows ]
	for (int i = 0; i < channels_histogram.size(); i++) {
		normalize(channels_histogram[i], channels_histogram[i], 0, histImage.rows, NORM_MINMAX, -1, Mat());
	}

	vector<Scalar> colors = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0, 0, 255)};

	// Draw for each channel
	for (int i = 1; i < hist_size; i++)
	{
		for (int j = 0; j < channels_histogram.size();j++) {
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(channels_histogram[j].at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(channels_histogram[j].at<float>(i))),
				colors[j], 2, 8, 0);
		}
	}

	return histImage;
}

void SaltAndPepper(Mat *image, float noise) {

	int n_channels = (*image).channels();
	cout << "Channels: " << n_channels << endl;

	int n_noise_pixels = (*image).size().width * (*image).size().height*noise;
	cout << "Number of Noise Pixels: " << n_noise_pixels << endl;

	for (int i = 0; i < n_noise_pixels; i++) {

		//random pixel
		int pixel_x = rand() % (*image).size().width;
		int pixel_y = rand() % (*image).size().height;

		int rnd = rand() % 2;
		int color;

		if (rnd == 1)		//black
			color = 0;
		else
			color = 255;	//white

		if (n_channels == 1) {

			//different color from random
			if ((*image).at<uchar>(pixel_x, pixel_y) != color)
				(*image).at<uchar>(pixel_x, pixel_y) = color;
			//same color as random
			else if (color == 255)
				(*image).at<uchar>(pixel_x, pixel_y) = 0;
			else if (color == 0)
				(*image).at<uchar>(pixel_x, pixel_y) = 255;
		}
		else if (n_channels == 3) {
			//different color from random
			if ((*image).at<Vec3b>(Point(pixel_x, pixel_y)) != Vec3b(color, color, color))
				(*image).at<Vec3b>(Point(pixel_x, pixel_y)) = Vec3b(color, color, color);
			//same color as random
			else if (color == 255)
				(*image).at<Vec3b>(Point(pixel_x, pixel_y)) = Vec3b(0, 0, 0);
			else if (color == 0)
				(*image).at<Vec3b>(Point(pixel_x, pixel_y)) = Vec3b(255, 255, 255);
		}
	}

}

void ApplyHistogramEqualization(Mat *image) {

	vector<Mat> channels;
	split(*image, channels);

	//apply histogram equalization on each channel
	vector<Mat> equalized_channels;

	for (int i = 0; i < channels.size(); i++) {
		equalized_channels.push_back(Mat());
		equalizeHist(channels[i], equalized_channels[i]);
	}

	Mat equalized;
	merge(equalized_channels, *image);
}

void ApplyClahe(Mat *image, float clip_image) {

	//split channels
	vector<Mat> channels;
	split(*image, channels);

	Ptr<CLAHE> clahe = createCLAHE(clip_image);

	//apply clahe in each channel
	vector<Mat> channels_clahe;
	for (int i = 0; i < channels.size(); i++) {
		channels_clahe.push_back(Mat());
		clahe->apply(channels[i], channels_clahe[i]);
	}

	//merge
	merge(channels_clahe, *image);
}
