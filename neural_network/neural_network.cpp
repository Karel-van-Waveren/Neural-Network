#include "stdafx.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "neural_network.h"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <experimental/filesystem>

typedef vector<string>::const_iterator myvector;
namespace fs = std::experimental::filesystem::v1;

Mat neural_network::get_descriptors(Mat img)
{
	Mat descriptors;
	Ptr<ORB> orb = ORB::create();
	Mat mask;
	vector<KeyPoint> kp = vector<KeyPoint>();
	orb->detectAndCompute(img, mask, kp, descriptors);
	return descriptors;
}

void neural_network::read_files(string pathImages, vector<string> & files) {
	for (auto & p : fs::directory_iterator(pathImages)) {
		files.push_back(p.path().string());
	}
}

void neural_network::read_images(myvector begin, myvector end, std::function<void(const std::string&, const cv::Mat&)> callback) {
	for (auto it = begin; it != end; it++) {
		string readfilename = *it;
		// Imread 1 --> color // Imread 0 --> grayscale
		Mat image = imread(readfilename, 1);
		if (image.empty()) {
			cout << "The image is empty! " << endl;
		}
		else {
			string classname = readfilename.substr(readfilename.find_last_of("\\") + 1, 3);
			Mat descriptors = this->get_descriptors(image);
			callback(classname, descriptors);
		}
	}
}
