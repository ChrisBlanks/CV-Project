#include "ImageProcessing.h"


void imageProcess::createPoints(void) {
	cv::Point pt1; //Create a point
	pt1.x = 20; pt1.y = 20;

	cv::Point pt2 = cv::Point(10,7);
	return;
}

void imageProcess::drawAtom(void) {
	srand(time(NULL)); //the seed is the number of seconds in computer time since Jan 1, 1970
	int width = 400;
	char atom_window[] = "Drawing: Atom"; //title of window

	cv::Mat atom_image = cv::Mat::zeros(width, width, CV_8UC3);
	cv::namedWindow(atom_window, cv::WINDOW_AUTOSIZE);

	for (int angle = -45; angle <= 90; angle += 45) {
		createEllipse(atom_image,angle,width); //creates ellipses in atom_image
	}

	createFilledCircle(atom_image,cv::Point(width/2,width/2),width);
	cv::imshow(atom_window,atom_image);
	cv::waitKey(0);
}

void imageProcess::createEllipse(cv::Mat input, double angle,int width) {
	int thickness = 2;
	int lineType = 8;
	
	// "rand()%10" will create a random integer and get the ones place value (range is 0 to 9)
	// The "255/9" factor will map the random integer range to the 0 to 255 range for BGR convention
	cv::ellipse(input,cv::Point(width/2,width/2),cv::Size(width/4,width/16),angle,0,360,
		cv::Scalar( (rand() % 10) * 255/9 , (rand() % 10) * 255 / 9, (rand() % 10) * 255 / 9),thickness,lineType);
	return;
}

void imageProcess::createFilledCircle(cv::Mat input,cv::Point center, int width) {
	cv::circle(input,center,width/32, cv::Scalar((rand() % 10) * 255 / 9, (rand() % 10) * 255 / 9, (rand() % 10) * 255 / 9),
		cv::FILLED,cv::LINE_8);
	return;
}