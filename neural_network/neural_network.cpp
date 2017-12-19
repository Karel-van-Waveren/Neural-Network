#include "stdafx.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "neural_network.h"
#include <opencv2/highgui.hpp>


Mat neural_network::get_descriptors(Mat img, Mat & descriptors)
{
	Ptr<ORB> orb = ORB::create();
	Mat mask;
	vector<KeyPoint> kp = vector<KeyPoint>();
	orb->detectAndCompute(img, mask, kp, descriptors);
	Mat img2;
	drawKeypoints(img, kp, img2);
	imshow("aardbei", img2);
	return descriptors;
}
