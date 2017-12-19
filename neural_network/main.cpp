// neural_network.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "neural_network.h"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main()
{
	string path = "pics/kleine.jpg";
	Mat image, gauss;
	image = imread(path);
	GaussianBlur(image, gauss, Size(0, 0), 2, 2);

	Mat descriptors;

	neural_network n_n = neural_network();
	n_n.get_descriptors(image, descriptors);

	waitKey(0);
    return 0;
}

