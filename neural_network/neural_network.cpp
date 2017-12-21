#include "stdafx.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "neural_network.h"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <experimental/filesystem>
#include <set>
#include <opencv2/ml.hpp>
#include <fstream>
#include <iomanip>

typedef vector<string>::const_iterator myvector;
namespace fs = experimental::filesystem::v1;

// get features using orb feature matching
Mat neural_network::get_features_orb(Mat img)
{
	Ptr<ORB> orb = ORB::create();
	vector<KeyPoint> kp;
	Mat features;
	orb->detectAndCompute(img, noArray(), kp, features);
	return features;
}

// get features using Accelerated KAZE feature matching
Mat neural_network::get_features_AKAZE(Mat img)
{
	Ptr<AKAZE> akaze = AKAZE::create();
	vector<KeyPoint> kp;
	Mat features;
	akaze->detectAndCompute(img, noArray(), kp, features);
	return features;
}

// read all files in path directory
void neural_network::read_files(string pathImages, vector<string> & files)
{
	for (auto & p : fs::directory_iterator(pathImages)) {
		files.push_back(p.path().string());
	}
}

// read all images and get features
void neural_network::read_images(myvector begin, myvector end, function<void(const string&, const Mat&)> callback)
{
	for (auto it = begin; it != end; it++) {
		string readfilename = *it;
		Mat image = imread(readfilename, 0);
		if (image.empty()) {
			std::cout << "The image is empty! " << std::endl;
			fs::remove(readfilename);
		}
		else {
			string classname = readfilename.substr(readfilename.find_last_of("\\") + 1, 5);
			Mat features = this->get_features_orb(image);
			//Mat features = this->get_features_AKAZE(image);
			callback(classname, features);
		}
	}
}

// read image from camera and get features
void neural_network::read_image_camera(VideoCapture camera, function<void(const Mat&)> callback)
{
	Mat image;
	camera >> image;
	if (image.empty()) {
		std::cout << "The image is empty! " << std::endl;
	}
	else {
		Mat features = this->get_features_orb(image);
		//Mat features = this->get_features_AKAZE(image);
		callback(features);
	}
	imshow(" ",image);
	waitKey(1);
}

// get class codes 
Mat neural_network::get_class_code(const set<string>& classes, const string& classname)
{
	Mat code = Mat::zeros(Size((int)classes.size(), 1), CV_32F);
	int index = get_class_id(classes, classname);
	code.at<float>(index) = 1;
	return code;
}

// get class id
int neural_network::get_class_id(const set<string>& classes, const string& classname)
{
	int index = 0;
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		if (*it == classname) break;
		++index;
	}
	return index;
}

// get class name
string neural_network::get_class_name(const set<string>& classes, const int classid)
{
	auto it = classes.begin(); 
	advance(it, classid);
	return *it;
}

// get a neural network and train it
Ptr<ml::ANN_MLP> neural_network::get_trained_neural_network(const Mat& train_samples, const Mat& train_responses)
{
	int network_input_size = train_samples.cols;
	int network_output_size = train_responses.cols;
	Ptr<ml::ANN_MLP> mlp = ml::ANN_MLP::create();
	vector<int> layer_sizes = { network_input_size, network_input_size / 2,
		network_output_size };
	mlp->setLayerSizes(layer_sizes);
	mlp->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
	mlp->train(train_samples, ml::ROW_SAMPLE, train_responses);
	return mlp;
}

// get the features in bag of words model
Mat neural_network::get_bow_features(FlannBasedMatcher& flann, const Mat& features, int vocabulary_size)
{
	Mat output_array = Mat::zeros(Size(vocabulary_size, 1), CV_32F);
	Mat features_CV_32S;
	features.convertTo(features_CV_32S, CV_32F);
	vector<DMatch> matches;

	flann.match(features_CV_32S, matches);

	for (size_t i = 0; i < matches.size(); i++)
	{
		int visual_word = matches[i].trainIdx;
		output_array.at<float>(visual_word)++;
	}
	return output_array;
}

// get the predicted class, optionally print out the predicted class if print_probability is set to 1
int neural_network::get_predicted_class(const Mat& predictions, set<string> classes, int print_probability)
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
	if (print_probability == 1) {
		if (max_prediction > 1) std::cout << "predicted class: " << this->get_class_name(classes, max_prediction_index) << " with " << max_prediction << " probability" << std::endl;
		else std::cout << "no object detected" << std::endl;
	}
	return max_prediction_index;
}

// get the confusion matrix 
vector<vector<int>> neural_network::get_confusion_matrix(Ptr<ml::ANN_MLP> mlp, const Mat& test_samples, const vector<int>& test_output_expected, set<string> & classes)
{
	Mat test_output;
	neural_network n_n = neural_network();

	mlp->predict(test_samples, test_output);
	vector<vector<int>> confusion_matrix(classes.size(), vector<int>(classes.size()));
	for (int i = 0; i < test_output.rows; i++)
	{
		int predicted_class = n_n.get_predicted_class(test_output.row(i),classes);
		int expected_class = test_output_expected.at(i);
		confusion_matrix[expected_class][predicted_class]++;
	}
	return confusion_matrix;
}

// print the confusion matrix
void neural_network::print_confusion_matrix(const vector<vector<int>>& confussion_matrix, const set<string> classes)
{
	std::cout << "  ";
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		std::cout << *it << setw(6);
	}
	std::cout << std::endl;
	for (size_t i = 0; i < confussion_matrix.size(); i++)
	{
		for (size_t j = 0; j < confussion_matrix[i].size(); j++)
		{
			std::cout << confussion_matrix[i][j] << setw(6);
		}
		std::cout << std::endl;
	}
}

// print the predicted class
void neural_network::print_predicted_class(Ptr<ml::ANN_MLP> mlp, const Mat& test_samples, set<string> & classes)
{
	Mat prediction;
	mlp->predict(test_samples, prediction);

	this->get_predicted_class(prediction.row(0),classes,1);
}

// get accuracy using the confusion matrix
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

// save models to path defined in the top of the main
void neural_network::save_models(string model_path, Ptr<ml::ANN_MLP> mlp, const Mat& vocabulary, const set<string>& classes)
{
	mlp->save(model_path + "mlp.yaml");

	FileStorage fs(model_path + "vocabulary.yaml", FileStorage::WRITE);
	fs << "vocabulary" << vocabulary;
	fs.release();

	neural_network n_n = neural_network();
	ofstream classes_output(model_path + "classes.txt");
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		classes_output << n_n.get_class_id(classes, *it) << "\t" << *it << std::endl;
	}
	classes_output.close();
}

// load models in the path defined in the top of the main
void neural_network::load_models(string model_path, Ptr<ml::ANN_MLP>& mlp, Mat& vocabulary, set<string>& classes)
{
	mlp = mlp->load(model_path + "mlp.yaml");

	FileStorage fs(model_path + "vocabulary.yaml", FileStorage::READ);
	fs["vocabulary"] >> vocabulary;
	fs.release();

	ifstream classes_input(model_path + "classes.txt");
	std::string line;
	while (getline(classes_input, line))
	{
		stringstream ss;
		ss << line;
		int index;
		string classname;
		ss >> index;
		ss >> classname;
		classes.insert(classname);
	}
}