#pragma once
#include <set>
#include <opencv2/ml.hpp>

namespace cv {
	class FlannBasedMatcher;
}

using namespace std;
using namespace cv;

class neural_network
{
public:
	Mat neural_network::get_descriptors(Mat img);
	// func: Reads images from filenames
	// 
	typedef vector<string>::const_iterator myvector;
	void neural_network::read_images(myvector begin, myvector end, function<void(const string&, const Mat&)> callback);
	void neural_network::read_files(string pathImages, vector<string> & files);
	Mat neural_network::get_class_code(const set<string>& classes, const string& classname);
	int neural_network::get_class_id(const set<string>& classes, const string& classname);
	Ptr<ml::ANN_MLP> neural_network::get_trainedNeural_network(const Mat& trainSamples, const Mat& trainResponses);
	Mat neural_network::get_bow_features(FlannBasedMatcher& flann, const Mat& descriptors,int vocabulary_size);
	};

	struct image_data
	{
		string class_name;
		Mat bow_features;
	};