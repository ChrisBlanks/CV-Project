#include "opencv_test_code.h"



void cb_func::imageOperations(void) {

	cv::Mat img = cv::imread((std::string)PICTURES_DIRECTORY + (std::string)TEST_IMAGE + (std::string)JPG_EXTENSION);
	cv::Mat grey_img = cv::imread((std::string)PICTURES_DIRECTORY + (std::string)TEST_IMAGE + (std::string)JPG_EXTENSION, cv::IMREAD_GRAYSCALE);

	if (img.empty() || grey_img.empty()) {
		std::cerr << "Could not find image.\n";
		return;
	}
	else {
		cv::imwrite((std::string)PICTURES_DIRECTORY + (std::string)TEST_IMAGE + "_grey_version" + (std::string)JPG_EXTENSION, grey_img);
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
			//searches for pixels that meet the BGR criteria & changes them to a new color
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

void cb_func::detectAndShow(cv::Mat frame,cv::CascadeClassifier face_casc, cv::CascadeClassifier eyes_casc) {
	cv::Mat gray_frame;
	cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(gray_frame,gray_frame); //this func is meant for grayscale images
	//makes histogram of intensities more even, which brings more contrast

	std::vector<cv::Rect> faces;
	face_casc.detectMultiScale(gray_frame, faces); //detects frontal face & puts results into faces vector

	for (size_t i = 0; i < faces.size(); i++) {
		cv::Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		cv::ellipse(frame, center, cv::Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, cv::Scalar(255, 0, 255), 4); 
		cv::Mat faceROI = gray_frame(faces[i]); //shrinks the region of interest for detecting eyes
		
		std::vector<cv::Rect> eyes;
		eyes_casc.detectMultiScale(faceROI,eyes); //detects eyes & puts results into eyes vector
		for (size_t j = 0; j < eyes.size(); j++) {
			cv::Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = ((eyes[j].width + eyes[j].height)*0.25);
			cv::circle(frame, eye_center, radius,cv::Scalar(255,0,0),4);
		}
	}
	cv::imshow("Real Time Detector",frame); //shows image with detected feature markers
}

void cb_func::findFace(void) {
	cv::CascadeClassifier face_cascade;
	cv::CascadeClassifier eyes_cascade;

	const std::string face_haarcascade = "/C++Libraries/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
	const std::string eyes_haarcascade = "/C++Libraries/opencv/build/etc/haarcascades/haarcascade_eye.xml";

	if (!face_cascade.load(face_haarcascade)) {
		std::cout << "Couldn't load face";
		return; }
	if (!eyes_cascade.load(eyes_haarcascade)) {
		std::cout << "Couldn't load eye";
		return; }

	cv::VideoCapture cap; //Opens default webcam

	if (!cap.open(0)) {
		std::cout << "Couldn't open camera";
		return;
	}

	cv::Mat frame;
	while (cap.read(frame)) {
		if (frame.empty()) {
			break; }

		cb_func::detectAndShow(frame,face_haarcascade,eyes_haarcascade);

		if (cv::waitKey(10) == ESC_CODE) { break; } //break if "esc" command is pressed
	}


	return;
}

void cb_func::implementMaskOp(void) {
	cv::Mat img = cv::imread((std::string)PICTURES_DIRECTORY + (std::string)TEST_IMAGE + (std::string)JPG_EXTENSION,cv::IMREAD_COLOR);
	cv::Mat out;

	if ( img.empty() ) {
		std::cerr << "Could not find image.\n";
		return;
	}

	cv::namedWindow("Input",cv::WINDOW_AUTOSIZE);
	cv::imshow("Input", img);

	cv::Mat kernel = (cv::Mat_<char>(3,3) << 0,-1,0,-1,5,-1,0,-1,0); //Makes a 3x3 kernel w/ mask values (image contrast enhancer)
	cv::filter2D(img, out,img.depth() ,kernel); //applies kernel filter to all pixels in 'img' source & places result into 'out'
	cv::namedWindow("Output",cv::WINDOW_AUTOSIZE);
	cv::imshow("Output", out);
	cv::waitKey(0); //displays window until keypress in the image window
	
	return;
}


//prompts user for alpha value
double cb_func::getOperatorValue(int ImageOperatorMode,double lower_bound = 0,double upper_bound = 1, double default_val = 0.5) {
	double operator_val;
	if (ImageOperatorMode == 0) { std::cout << "\nPlease, enter a value for alpha. The range is " << lower_bound << "-" << upper_bound << " :\n>>>"; }
	else if(ImageOperatorMode == 1){ std::cout << "\nPlease, enter a value for beta. The range is " << lower_bound << "-" << upper_bound << " :\n>>>"; }
	else if (ImageOperatorMode == 2) { std::cout << "\nPlease, enter a value for gamma. The range is " << lower_bound << "-" << upper_bound << " :\n>>>"; }
	std::cin >> operator_val;
	if (operator_val > lower_bound && operator_val < upper_bound) {
		std::cout << "\nAcceptable alpha value.\n";
	}
	else { operator_val = default_val; std::cout << "\nDefault value used: " << default_val << ".\n"; }
	return operator_val;
}


//blends two images to create a cross-dissolve effect by using the linear blend operator
void cb_func::blendImages(void) {
	double alpha = cb_func::getOperatorValue(cb_func::ALPHA_MODE);
	double beta = 1 - alpha;

	//loads pictures of runners
	cv::Mat out_blend; //blended output image

	cv::Mat src1 = cv::imread((std::string)PICTURES_DIRECTORY + (std::string)TEST_IMAGE_2 + (std::string)JPG_EXTENSION, cv::IMREAD_COLOR);
	cv::Mat src2 = cv::imread((std::string)PICTURES_DIRECTORY + (std::string)TEST_IMAGE_3 + (std::string)JPG_EXTENSION, cv::IMREAD_COLOR);
	cv::namedWindow("Image 1",cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Image 2", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);

	cv::resize(src1,src1,cv::Size(300,150));
	cv::resize(src2, src2,cv::Size(300,150));

	cv::imshow("Image 1", src1);
	cv::imshow("Image 2", src2);

	//blends images using: g(x) = f1_val_pix*beta + alpha*f2_val_pix + gamma
	//f1 = src1 ; f2 = src2 ; out_blend = g(x) 
	cv::addWeighted(src1,alpha,src2,beta,0,out_blend);// zeros out the gamma variable
	// When alpha > 0.5, second image is shown more. Vice versa for when alpha < 0.5
	cv::imshow("Output", out_blend);
	cv::waitKey(0);
	return;
}


static void cb_func::brightness_trackbar(int value, void* userdata) {
	//callback function for slider
	double contrast = *(double *)userdata; //referenced value needed for calculations
	double brightness = value; //value received from slider position

	cv::Mat source = cv::imread(cb_func::PICTURES_DIRECTORY + cb_func::TEST_IMAGE_2 + cb_func::JPG_EXTENSION);
	if (source.empty()) { return; }
	cv::Mat output = cv::Mat::zeros(source.size(), source.type()); //zeroed matrix of same size & type
	cb_func::editContrastAndBrightness(source, output,contrast, brightness);
	cv::imshow("Edited Image",output);
}


static void cb_func::contrast_trackbar(int value, void*userdata) {
	//callback function for slider
	double contrast = value;
	double brightness = *(double *)userdata;
	
	cv::Mat source = cv::imread(cb_func::PICTURES_DIRECTORY + cb_func::TEST_IMAGE_2 + cb_func::JPG_EXTENSION);
	if (source.empty()) { return; }

	cv::Mat output = cv::Mat::zeros(source.size(), source.type()); //zeroed matrix of same size & type
	cb_func::editContrastAndBrightness(source, output, contrast, brightness);
	
	cv::imshow("Edited Image", output);
}

void cb_func::startTrackbarForContrastAndBrightness(void) {
	double constrast_slider =1, brightness_slider=1;
	int placeholder = 1; //what value the slider is initialized at
	
	const int contrast_slide_max = 3; 
	const int brightness_slide_max = 100;

	cv::Mat source = cv::imread(cb_func::PICTURES_DIRECTORY + cb_func::TEST_IMAGE_2 + cb_func::JPG_EXTENSION);
	if (source.empty()) { return; }
	cv::imshow("Original Image", source);
	cv::namedWindow("Edited Image", cv::WINDOW_AUTOSIZE);

	char ContrastTrackbarName[50], BrightnessTrackbarName[50];

	sprintf_s(ContrastTrackbarName,"Contrast x %d",contrast_slide_max); //formats string into name
	sprintf_s(BrightnessTrackbarName, "Brightness x %d", brightness_slide_max);
	
	cv::createTrackbar(ContrastTrackbarName,"Edited Image"
		,&placeholder,contrast_slide_max, contrast_trackbar,(void*)&brightness_slider);

	cv::createTrackbar(BrightnessTrackbarName, "Edited Image"
		, &placeholder, brightness_slide_max, brightness_trackbar,(void*)&constrast_slider);

	cv::waitKey(0);
	return;
}

void cb_func::editContrastAndBrightness(cv::Mat src, cv::Mat output,double alpha, double beta) {
	//Note: Beta can make an image brighter & alpha spreads the color levels (either compresses or decompresses the histogram of values),
	// which can affect the contrast.
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			for (int c = 0; c < src.channels(); c++) {
				// g(x) = alpha*pix_val + beta     (for each pixel's channel values)
				output.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(alpha*src.at<cv::Vec3b>(y, x)[c] + beta);
				//saturate_cast<uchar> clips values if they exceed the size of the final type
			}
		}
	}

	return;
}

void cb_func::changeContrastAndBrightness(int OperationMode = 0) {
	
	if (OperationMode == cb_func::TRACKBAR_MODE) {
		startTrackbarForContrastAndBrightness();
	}
	else {
		cv::Mat src = cv::imread(cb_func::PICTURES_DIRECTORY + cb_func::TEST_IMAGE_2 + cb_func::JPG_EXTENSION);
		if (src.empty()) { return; }
		cv::Mat out = cv::Mat::zeros(src.size(), src.type()); //zeroed matrix of same size & type
		double alpha = cb_func::getOperatorValue(cb_func::ALPHA_MODE, 1, 3, 1); // range: 1-3; default: 1
		double beta = cb_func::getOperatorValue(cb_func::BETA_MODE, 0, 100, 0); // range: 0-100; default: 0

		editContrastAndBrightness(src, out, alpha, beta);
		// Note: Does the same thing, but faster:    src.convertTo(out, -1, alpha, beta);

		cv::imshow("Source", src);
		cv::imshow("Output", out);
		cv::waitKey(0);
	}

	return;
}

//can be used to correct the brightness of an image using a non-linear transformation:
// Out = 255 * (IN/255)^gamma
void cb_func::performGammaCorrection(void) {
	double gamma = cb_func::getOperatorValue(cb_func::GAMMA_MODE,0,25,0.4);
	cv::Mat src = cv::imread(cb_func::PICTURES_DIRECTORY + cb_func::TEST_IMAGE_2 + cb_func::JPG_EXTENSION,cv::IMREAD_COLOR);
	if (src.empty()) { return; }

	cv::Mat lookUpTable(1,256,CV_8U); //1 by 256 vector with 8-bit unsigned data
	uchar* p = lookUpTable.ptr();
	for (int i=0; i < 256; ++i) {
		p[i] = cv::saturate_cast<uchar>(pow((i/255),gamma)*255.0); //performs gamma operation on table values
	}
	cv::Mat out = src.clone();
	cv::LUT(src, lookUpTable, out); //copies the pre-calculated values in lookUpTabl into the "out" matrix using input indices

	cv::imshow("Input",src);
	cv::imshow("Output",out);
	cv::waitKey(0);
	return;
}


//Performs a dft on an input image
void cb_func::performDiscreteFourierTransform(void) {

	cv::Mat padded; //origninal src will be placed in the center of padded matrix
	cv::Mat src = cv::imread(cb_func::PICTURES_DIRECTORY + cb_func::TEST_IMAGE_2 + cb_func::JPG_EXTENSION, cv::IMREAD_GRAYSCALE);
	if (src.empty()) { return; }

	int m = cv::getOptimalDFTSize(src.rows); //gets optimal size for DFT
	int n = cv::getOptimalDFTSize(src.cols); //image size will become a multiple of two, three, and five
	cv::copyMakeBorder(src,padded,0,m-src.rows,0,n-src.cols,cv::BORDER_CONSTANT,cv::Scalar::all(0)); //expands border of image

	cv::Mat planes[] = {cv::Mat_<float>(padded),cv::Mat::zeros(padded.size(),CV_32F)};
	cv::Mat complexSrc; //Fourier transform deals with imaginary numbers

	cv::merge(planes, 2, complexSrc); //creates a multi-channel matrix (2 channels in this case)
	cv::dft(complexSrc, complexSrc); //performs dft

	cv::split(complexSrc,planes); //planes[0] = real part ; planes[1] = imaginary
	cv::magnitude(planes[0], planes[1], planes[0]); //magnitude = sqrt(dimen_1^2 + dimen_2^2); stored in 0th arg
	cv::Mat magnitudeSrc = planes[0]; 

	magnitudeSrc += cv::Scalar::all(1); // formula: M = log(1+ M)
	cv::log(magnitudeSrc, magnitudeSrc); //switches to log scale because range of fourier coefficients too large

	magnitudeSrc = magnitudeSrc(cv::Rect(0,0,magnitudeSrc.cols & -2,magnitudeSrc.rows & -2)); //crop if odd number of rows or columns
	
	int cx = magnitudeSrc.cols / 2;
	int cy = magnitudeSrc.rows / 2;

	cv::Mat q0(magnitudeSrc, cv::Rect(0, 0, cx, cy)); //top left quadrant
	cv::Mat q1(magnitudeSrc, cv::Rect(cx, 0, cx, cy)); //top right 
	cv::Mat q2(magnitudeSrc, cv::Rect(0, cy, cx, cy)); //bottom left
	cv::Mat q3(magnitudeSrc, cv::Rect(cx, cy, cx, cy)); //bottom right

	cv::Mat temp;

	q0.copyTo(temp); //swap top left with bottom right
	q3.copyTo(q0);
	temp.copyTo(q3);

	q1.copyTo(temp); //swap top right with bottom left
	q2.copyTo(q1);
	temp.copyTo(q2);

	int alpha = 0; int beta = 1;
	cv::normalize(magnitudeSrc, magnitudeSrc, alpha,beta, cv::NORM_MINMAX);
	//transforms into a viewable image (float values between 0 & 1)

	cv::imshow("Input", src);
	cv::imshow("Spectrum magnitude", magnitudeSrc); //will show an image that is representative of geometrical orientation
	cv::waitKey(0);

	return;
}


