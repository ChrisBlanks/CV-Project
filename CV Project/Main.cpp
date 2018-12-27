//Programmer: Chris Blanks
#include "opencv_test_code.h"

int main(int argc, char* argv[]) {

	std::vector<char*> arguments;
	if (argc > 1) {
		for (int i = 0; i < argc; i++) {
			arguments.push_back(argv[i]);
		}
	}

	cb_func::changeContrastAndBrightness(cb_func::TRACKBAR_MODE);
	//cb_func::performDiscreteFourierTransform();
	//cb_func::performGammaCorrection();
	//cb_func::changeContrastAndBrightness();
	//cb_func::blendImages();
	//cb_func::implementMaskOp();
	//cb_func::findFace();
	//cb_func::webcamTest();
	//cb_func::imageOperations();

	return 0;
}


