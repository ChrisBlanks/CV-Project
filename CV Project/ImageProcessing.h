#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <cstdlib>
#include <ctime>


namespace imageProcess{
	void createPoints(void);
	void drawAtom(void);
	void createEllipse(cv::Mat input,double angle,int width);
	void createFilledCircle(cv::Mat input, cv::Point center,int width);
}