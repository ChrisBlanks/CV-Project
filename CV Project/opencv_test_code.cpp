#include "opencv_test_code.h"


using namespace cb_func;

void cb_func::imageOperations(void) {

	cv::Mat img = cv::imread((std::string)PICTURES_DIRECTORY + (std::string)TEST_IMAGE + (std::string)TEST_IMAGE_EXTENSION);
	cv::Mat grey_img = cv::imread((std::string)PICTURES_DIRECTORY + (std::string)TEST_IMAGE + (std::string)TEST_IMAGE_EXTENSION, cv::IMREAD_GRAYSCALE);

	if (img.empty() || grey_img.empty()) {
		std::cerr << "Could not find image.\n";
		return;
	}
	else {
		cv::imwrite((std::string)PICTURES_DIRECTORY + (std::string)TEST_IMAGE + "_grey_version" + (std::string)TEST_IMAGE_EXTENSION, grey_img);
		cv::imshow("Title: Image", img);
		cv::imshow("Title: Greyscale Image", grey_img);
		cv::waitKey(0); //displays window until keypress in the image window
	}

	for (int y = 0; y < img.size().height / 50; y++) {
		for (int x = 0; x < img.size().width / 50; x++) {
			cv::Vec3b pixel_intensity = img.at<cv::Vec3b>(y, x);
			uchar red = pixel_intensity.val[0];
			uchar green = pixel_intensity.val[1];
			uchar blue = pixel_intensity.val[2];

			std::cout << "RGB Values: " << (int)red << ", " << (int)green << ", " << (int)blue << std::endl;


		}
	}

}

//edits the original frame
void cb_func::editFrame(cv::Mat frame) {

	for (int y = 0; y < frame.rows; y++) {
		for (int x = 0; x < frame.cols; x++) {
			cv::Vec3b pix_colors = frame.at<cv::Vec3b>(y, x);

			if (pix_colors[0] < B_OLD && pix_colors[1] > G_OLD && pix_colors[2] < R_OLD) {
				pix_colors[0] = R_NEW;
				pix_colors[1] = G_NEW;
				pix_colors[2] = B_NEW;
			}

			frame.at<cv::Vec3b>(y, x) = pix_colors;
		}
	}

	return; 
}


void cb_func::webcamTest(void) {
	cv::VideoCapture cap; //Opens default webcam
	if (!cap.open(0)) {
		return;
	}

	//run forever
	for (;;) {
		cv::Mat frame;
		cap >> frame;

		if (frame.empty()) break; //break if frame isn't captured
		cb_func::editFrame(frame);
		cv::imshow("Webcam Stream", frame); //display each frame indefinitely
		if (cv::waitKey(10) == ESC_CODE) break; //break if ESC is pressed
	}

	cap.release();

	return;
}


