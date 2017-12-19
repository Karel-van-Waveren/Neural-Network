#include "stdafx.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "neural_network.h"



Mat neural_network::get_descriptors(Mat img, vector<KeyPoint> & kp)
{
	Ptr<ORB> orb = ORB::create();
	orb->detect(img, kp);
	orb->compute(img, kp);
	return img;
}
