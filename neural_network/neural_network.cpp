#include "stdafx.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "neural_network.h"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <experimental/filesystem>
#include <set>
#include <opencv2/ml.hpp>

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

void neural_network::read_images(myvector begin, myvector end, std::function<void(const std::string&, const Mat&)> callback) {
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

Mat neural_network::get_class_code(const set<std::string>& classes, const std::string& classname)
{
	Mat code = Mat::zeros(Size((int)classes.size(), 1), CV_32F);
	int index = get_class_id(classes, classname);
	code.at<float>(index) = 1;
	return code;
}

int neural_network::get_class_id(const std::set<std::string>& classes, const std::string& classname)
{
	int index = 0;
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		if (*it == classname) break;
		++index;
	}
	return index;
}

Ptr<ml::ANN_MLP> neural_network::get_trainedNeural_network(const Mat& train_samples, const Mat& train_responses)
{
	int network_input_size = train_samples.cols;
	int network_output_size = train_responses.cols;
	Ptr<ml::ANN_MLP> mlp = ml::ANN_MLP::create();
	std::vector<int> layer_sizes = { network_input_size, network_input_size / 2,
		network_output_size };
	mlp->setLayerSizes(layer_sizes);
	mlp->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
	mlp->train(train_samples, ml::ROW_SAMPLE, train_responses);
	return mlp;
}
