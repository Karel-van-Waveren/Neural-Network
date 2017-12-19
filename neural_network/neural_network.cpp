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

Mat neural_network::get_bow_features(FlannBasedMatcher& flann, const Mat& descriptors, int vocabulary_size)
{
	Mat output_array = Mat::zeros(Size(vocabulary_size, 1), CV_32F);
	Mat descriptors_CV_32S;
	descriptors.convertTo(descriptors_CV_32S, CV_32F);
	vector<cv::DMatch> matches;

	flann.match(descriptors_CV_32S, matches);

	for (size_t i = 0; i < matches.size(); i++)
	{
		int visual_word = matches[i].trainIdx;
		output_array.at<float>(visual_word)++;
	}
	return output_array;
}

int neural_network::get_predicted_class(const Mat& predictions)
{
	float max_prediction = predictions.at<float>(0);
	float max_prediction_index = 0;
	const float* ptr_predictions = predictions.ptr<float>(0);
	for (int i = 0; i < predictions.cols; i++)
	{
		float prediction = *ptr_predictions++;
		if (prediction > max_prediction)
		{
			max_prediction = prediction;
			max_prediction_index = i;
		}
	}
	return max_prediction_index;
}

vector<vector<int>> neural_network::get_confusion_matrix(Ptr<ml::ANN_MLP> mlp, const Mat& test_samples, const vector<int>& test_output_expected)
{
	Mat test_output;
	neural_network n_n = neural_network();

	mlp->predict(test_samples, test_output);
	vector<vector<int>> confusion_matrix(2, vector<int>(2));
	for (int i = 0; i < test_output.rows; i++)
	{
		int predicted_class = n_n.get_predicted_class(test_output.row(i));
		int expected_class = test_output_expected.at(i);
		confusion_matrix[expected_class][predicted_class]++;
	}
	return confusion_matrix;
}

void neural_network::print_confusion_matrix(const vector<vector<int>>& confussion_matrix, const set<string> classes)
{
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		cout << *it << " ";
	}
	cout << endl;
	for (size_t i = 0; i < confussion_matrix.size(); i++)
	{
		for (size_t j = 0; j < confussion_matrix[i].size(); j++)
		{
			cout << confussion_matrix[i][j] << " ";
		}
		cout << endl;
	}
}

float neural_network::get_accuracy(const vector<vector<int>>& confusion_matrix)
{
	int hits = 0;
	int total = 0;
	for (size_t i = 0; i < confusion_matrix.size(); i++)
	{
		for (size_t j = 0; j < confusion_matrix.at(i).size(); j++)
		{
			if (i == j) hits += confusion_matrix.at(i).at(j);
			total += confusion_matrix.at(i).at(j);
		}
	}
	return hits / (float)total;
}