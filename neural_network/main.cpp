// neural_network.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main()
{
	string path = "pics/aardbei.jpg";
	Mat image, gauss;
	image = imread(path);
	GaussianBlur(image, gauss, Size(0, 0), 2, 2);
	imshow("gauss", gauss);
	waitKey(0);
    return 0;
}

