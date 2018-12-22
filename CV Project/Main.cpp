//Programmer: Chris Blanks
#include "opencv_test_code.h"

int main(int argc, char* argv[]) {

	std::vector<char*> arguments;
	if (argc > 1) {
		for (int i = 0; i < argc; i++) {
			arguments.push_back(argv[i]);
		}
	}
	cb_func::findFace();
	//cb_func::webcamTest();
	//cb_func::imageOperations();

	return 0;
}


