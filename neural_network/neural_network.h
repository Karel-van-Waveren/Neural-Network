#pragma once
#include <set>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;

class neural_network
{
public:
	Mat neural_network::get_descriptors(Mat img);
	// func: Reads images from filenames
	// 
	typedef vector<string>::const_iterator myvector;
	void neural_network::read_images(myvector begin, myvector end, std::function<void(const std::string&, const cv::Mat&)> callback);
	void neural_network::read_files(string pathImages, vector<string> & files);
	Mat neural_network::get_class_code(const set<std::string>& classes, const std::string& classname);
	int neural_network::get_class_id(const std::set<std::string>& classes, const std::string& classname);
	Ptr<ml::ANN_MLP> neural_network::get_trainedNeural_network(const Mat& trainSamples, const Mat& trainResponses);
	};

	struct image_data
	{
		std::string class_name;
		cv::Mat bow_features;
	};