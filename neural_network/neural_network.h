#pragma once
#include <set>
#include <opencv2/ml.hpp>

namespace cv {
	class VideoCapture;
	class FlannBasedMatcher;
}

using namespace std;
using namespace cv;

class neural_network
{
public:
	typedef vector<string>::const_iterator myvector;
	void neural_network::read_images(myvector begin, myvector end, function<void(const string&, const Mat&)> callback);
	void neural_network::read_image_camera(VideoCapture camera, function<void(const Mat&)> callback);
	void neural_network::read_files(string pathImages, vector<string> & files);
	Mat neural_network::get_class_code(const set<string>& classes, const string& classname);
	int neural_network::get_class_id(const set<string>& classes, const string& classname);
	Ptr<ml::ANN_MLP> neural_network::get_trained_neural_network(const Mat& trainSamples, const Mat& trainResponses);
	Mat neural_network::get_bow_features(FlannBasedMatcher& flann, const Mat& features, int vocabulary_size);
	vector<vector<int>> neural_network::get_confusion_matrix(Ptr<ml::ANN_MLP> mlp, const Mat& test_samples, const vector<int>& test_output_expected, set<string> & classes);
	void neural_network::print_confusion_matrix(const vector<vector<int>>& confussion_matrix, const set<string> classes);
	float neural_network::get_accuracy(const vector<vector<int>>& confusion_matrix);
	void neural_network::save_models(string model_path, Ptr<ml::ANN_MLP> mlp, const Mat& vocabulary, const set<string>& classes);
	void neural_network::load_models(string model_path, Ptr<ml::ANN_MLP>& mlp, Mat& vocabulary, set<string>& classes);
	void neural_network::print_predicted_class(Ptr<ml::ANN_MLP> mlp, const Mat& test_samples, set<string> & classes);
	string neural_network::get_class_name(const set<string>& classes, const int classid);
private:
	Mat neural_network::get_features_orb(Mat img);
	Mat neural_network::get_features_AKAZE(Mat img);
	int neural_network::get_predicted_class(const Mat& predictions, set<string> classes = set<string>(), int print_probability = 0);
};

struct image_data
{
	string class_name;
	Mat bow_features;
};