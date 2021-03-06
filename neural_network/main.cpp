// neural_network.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "neural_network.h"
#include <iostream>
#include <string>
#include <set>
#include <opencv2/features2d.hpp>
#include <filesystem>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;
using namespace ml;

// location for the training image set
string training_image_path = "pics/singular/";
// location for the testing image set
string testing_image_path = "pics/singular/";
// location where trained machines will be saved
string model_path = training_image_path +"trained machines/";
// ratio of amount of training images to be used 0.0f ... 1.0f
float train_ratio = 1.0f;
// ratio of amount of test images to be used 0.0f ... 1.0f
float test_ratio = 0.3f;
// Amount of neurons in the input layer of the network
int neural_network_input_size = 512;

int train_neural_network()
{

	neural_network n_n = neural_network();
	vector<string> training_set;
	int network_input_size = 512;
	Mat feature_set;
	vector<image_data*> features_metadata;
	set<string> classes;

	std::cout << "loading training set..." << std::endl;
	n_n.read_files(training_image_path, training_set);

	random_shuffle(training_set.begin(), training_set.end());
	std::cout << "getting features from " << round(training_set.size()*train_ratio) << " images" << std::endl;
	
	n_n.read_images(training_set.begin(), training_set.begin() + (size_t)(training_set.size() * train_ratio), [&](const string& classname, const Mat& features)
	{
		classes.insert(classname);
		feature_set.push_back(features);
		image_data* data = new image_data;
		data->class_name = classname;
		data->bow_features = Mat::zeros(Size(network_input_size, 1), CV_32F);
		for (int j = 0; j < features.rows; j++)
		{
			features_metadata.push_back(data);
		}
	});

	std::cout << "quanitizing features" << std::endl;
	Mat labels;
	Mat vocabulary;
	Mat feature_set_CV_32F;
	feature_set.convertTo(feature_set_CV_32F, CV_32F);
	kmeans(feature_set_CV_32F, network_input_size, labels, TermCriteria(TermCriteria::EPS +
		TermCriteria::MAX_ITER, 10, 0.01), 1, KMEANS_PP_CENTERS, vocabulary);
	feature_set.release();

	int* ptr_labels = (int*)(labels.data);
	int size = labels.rows * labels.cols;
	for (int i = 0; i < size; ++i)
	{
		int label = *ptr_labels++;
		image_data* data = features_metadata[i];
		data->bow_features.at<float>(label)++;
	}

	std::cout << "normalizing features" << std::endl;
	Mat train_samples;
	Mat train_responses;
	set<image_data*> unique_metadata(features_metadata.begin(), features_metadata.end());
	for (auto it = unique_metadata.begin(); it != unique_metadata.end(); )
	{
		image_data* data = *it;
		Mat normalizedHist;
		normalize(data->bow_features, normalizedHist, 0, data->bow_features.rows, NORM_MINMAX, -1, Mat());
		train_samples.push_back(normalizedHist);
		train_responses.push_back(n_n.get_class_code(classes, data->class_name));
		delete *it;
		++it;
	}

	std::cout << "training neural network" << std::endl;
	Ptr<ml::ANN_MLP> mlp = n_n.get_trained_neural_network(train_samples, train_responses);
	train_samples.release();
	train_responses.release();

	std::cout << "training flann" << std::endl;
	FlannBasedMatcher flann;
	flann.add(vocabulary);
	flann.train();

	std::cout << "save machine? [Y/n]" << std::endl;
	string msg;
	std::cin >> msg;
	if (msg == "Y" || msg == "y")
	{
		std::cout << "saving..." << std::endl;
		n_n.save_models(model_path, mlp, vocabulary, classes);
	}

	return 0;
}

int test_neural_network()
{
	std::cout << "testing neural network using training set" << std::endl;
	neural_network n_n = neural_network();
	vector<string> test_set;
	int network_input_size = 512;
	set<string> classes;

	Ptr<ml::ANN_MLP> mlp;
	Mat vocabulary;

	std::cout << "loading neural network..." << std::endl;
	n_n.load_models(model_path, mlp, vocabulary, classes);

	std::cout << "training flann" << std::endl;
	FlannBasedMatcher flann;
	flann.add(vocabulary);
	flann.train();

	std::cout << "loading test set..." << std::endl;
	n_n.read_files(testing_image_path, test_set);
	random_shuffle(test_set.begin(), test_set.end());
	std::cout << "getting features from " << round(test_set.size()*test_ratio) << " images" << std::endl;

	Mat test_samples;
	vector<int> test_output_expected;
	n_n.read_images(test_set.begin(), test_set.begin() + (size_t)(test_set.size() * test_ratio), [&](const string& classname, const Mat& features)
	{
		Mat bow_features = n_n.get_bow_features(flann, features, network_input_size);
		normalize(bow_features, bow_features, 0, bow_features.rows, NORM_MINMAX, -1, Mat());
		test_samples.push_back(bow_features);
		test_output_expected.push_back(n_n.get_class_id(classes, classname));
	});

	vector<vector<int>> confusion_matrix = n_n.get_confusion_matrix(mlp, test_samples, test_output_expected, classes);

	std::cout << "confusion matrix: " << std::endl;
	n_n.print_confusion_matrix(confusion_matrix, classes);
	std::cout << "accuracy: " << n_n.get_accuracy(confusion_matrix) << std::endl;

	return 0;
}

int test_neural_network_camera()
{
	std::cout << "testing neural network using the camera" << std::endl;
	neural_network n_n = neural_network();
	set<string> classes;
	vector<string> files;
	Ptr<ml::ANN_MLP> mlp;
	Mat vocabulary;

	std::cout << "loading neural network..." << std::endl;
	n_n.load_models(model_path, mlp, vocabulary, classes);

	std::cout << "training flann" << std::endl;
	FlannBasedMatcher flann;
	flann.add(vocabulary);
	flann.train();


	VideoCapture camera = VideoCapture(1);

	Mat test_samples;
	std::cout << "starting test" << std::endl;
	while(true)
	{
		test_samples.release();
	n_n.read_image_camera(camera, [&](const Mat& features)
	{
		Mat bow_features = n_n.get_bow_features(flann, features, neural_network_input_size);
		normalize(bow_features, bow_features, 0, bow_features.rows, NORM_MINMAX, -1, Mat());
		test_samples.push_back(bow_features);
	});
	n_n.print_predicted_class(mlp, test_samples, classes);
	}

	return 0;
}

int main()
{
	string msg;	

	std::cout << "do you want to train a neural network? [Y/n]" << std::endl;
	cin >> msg;
	if (msg == "Y" || msg == "y")train_neural_network();

	std::cout << "do you want to test a neural network? [Y/n]" << std::endl;
	cin >> msg;
	if (msg == "Y" || msg == "y")
	{
		std::cout << "do you want to test with a camera? [Y/n]" << std::endl;
		cin >> msg;
		if (msg == "Y" || msg == "y") test_neural_network_camera();
		else test_neural_network();
	}

	system("pause");
	return 0;
}

