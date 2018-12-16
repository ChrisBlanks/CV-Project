#pragma once

//standard library dependencies
#include <vector>
#include <iostream>
#include <string>

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



namespace cb_func {

	const std::string PICTURES_DIRECTORY = "C:/Users/Cabla/Pictures" ;
	const std::string TEST_IMAGE= "/8-bit";
	const std::string TEST_IMAGE_EXTENSION = ".jpg";

	constexpr int ESC_CODE = 27;

	constexpr int R_OLD = 130;
	constexpr int G_OLD = 120;
	constexpr int B_OLD = 130;

	constexpr int R_NEW = 198;
	constexpr int G_NEW = 80;
	constexpr int B_NEW = 178;

	//functions for testing OpenCV library

	void imageOperations(void);
	void webcamTest(void);
	cv::Mat editFrame(cv::Mat frame);

}