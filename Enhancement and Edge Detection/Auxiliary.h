#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>
#include<conio.h>	

using namespace std;
using namespace cv;

Mat GetHistogram(Mat image);

void SaltAndPepper(Mat *image, float noise);

void ApplyHistogramEqualization(Mat *image);

void ApplyClahe(Mat *image, float clip_image);

void FindZeroCrossings(Mat& laplacian, Mat& zero_crossings);
