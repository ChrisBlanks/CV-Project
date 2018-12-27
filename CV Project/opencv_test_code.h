#pragma once

//standard library dependencies
#include <vector>
#include <iostream>
#include <string>
#include <functional>

//OpenCV dependencies
#include <opencv2/opencv.hpp>
#include <opencv2/world.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/fast_math.hpp>



namespace cb_func {

	const std::string PICTURES_DIRECTORY = "C:/Users/Cabla/Pictures" ;
	const std::string TEST_IMAGE= "/8-bit";
	const std::string TEST_IMAGE_2 = "/sequence1";
	const std::string TEST_IMAGE_3 = "/sequence2";
	const std::string JPG_EXTENSION = ".jpg";
	const std::string PNG_EXTENSION = ".png";

	constexpr int ESC_CODE = 27;

	constexpr int R_OLD = 130;
	constexpr int G_OLD = 120;
	constexpr int B_OLD = 130;

	constexpr int R_NEW = 198;
	constexpr int G_NEW = 80;
	constexpr int B_NEW = 178;

	enum ImageOperatorModes {ALPHA_MODE,BETA_MODE,GAMMA_MODE, TRACKBAR_MODE}; // used in getOperatorValue

	//functions for testing OpenCV library

	void imageOperations(void);
	void webcamTest(void);
	void editFrame(cv::Mat frame);
	void findFace(void);
	void detectAndShow(cv::Mat frame, cv::CascadeClassifier face, cv::CascadeClassifier eyes);
	void implementMaskOp(void);
	void blendImages(void);
	void brightness_trackbar(int value, void* userdata);
	void contrast_trackbar(int value, void* userdata);
	void startTrackbarForContrastAndBrightness(void);
	void editContrastAndBrightness(cv::Mat src, cv::Mat output, double alpha, double beta);
	void changeContrastAndBrightness(int OperationMode);
	void performGammaCorrection(void);
	void performDiscreteFourierTransform(void);

	double getOperatorValue(int ImageOperatorMode, double lower_bound, double upper_bound, double default_val);
}