#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void readImage(const char* file, Mat& result) {
	result = imread(file, CV_LOAD_IMAGE_COLOR);
}

void display(const char* windowTitle, Mat& img) {
	namedWindow(windowTitle, WINDOW_AUTOSIZE);
	imshow(windowTitle, img);
	waitKey(0);
}

void convertToGrayImg(Mat& img, Mat& result) {
	cvtColor(img, result, CV_BGR2GRAY);
}

void subtractIntensityImage(Mat& bgrImg, Mat& grayImg, Mat& result) {
	//bgrImg - 0.5*grayImg
	result = bgrImg.clone();
	for(int r=0; r < bgrImg.rows; r++) {
		for(int c=0; c < bgrImg.cols; c++) {
			result.at<Vec3b>(r,c)[0] = std::max( bgrImg.at<Vec3b>(r,c)[0] - 0.5*grayImg.at<uchar>(r,c),0.); //blue
			result.at<Vec3b>(r,c)[1] = std::max( bgrImg.at<Vec3b>(r,c)[1] - 0.5*grayImg.at<uchar>(r,c),0.); //green
			result.at<Vec3b>(r,c)[2] = std::max( bgrImg.at<Vec3b>(r,c)[2] - 0.5*grayImg.at<uchar>(r,c),0.);	//red		
		}
	}
}

void pixelwiseSubtraction(Mat& bgrImg, Mat& grayImg, Mat& result) {

	int channels = bgrImg.channels();

	int nRows = bgrImg.rows;
	int nCols = bgrImg.cols * channels;

	result = bgrImg.clone();	

	if (bgrImg.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}

	for( int i = 0; i < nRows; i++) {
		int cols_grayImg = 0;
		for (int j = 0; j < nCols; j=j+3) {
			result.ptr<uchar>(i)[j] = std::max(bgrImg.ptr<uchar>(i)[j]-0.5*grayImg.ptr<uchar>(i)[cols_grayImg],0.);
			result.ptr<uchar>(i)[j+1] = std::max(bgrImg.ptr<uchar>(i)[j+1]-0.5*grayImg.ptr<uchar>(i)[cols_grayImg],0.);
			result.ptr<uchar>(i)[j+2] = std::max(bgrImg.ptr<uchar>(i)[j+2]-0.5*grayImg.ptr<uchar>(i)[cols_grayImg],0.);
			cols_grayImg++;
		}
	}
}

void extractPatch(Mat& img, Mat& result) {
	Rect myROI((int) img.cols/2-8, (int) img.rows/2-8, 16, 16);
	result = img.clone();
	result = result(myROI);
}

void copyPatchToRandomLocation(Mat& img, Mat& patch, Mat& result) {
	unsigned int random1 = std::rand()%(img.rows-16);
	unsigned int random2 = std::rand()%(img.cols-16);

	result = img.clone();	
	int i,j = 0;
	for(int r=random1; r < random1+16; r++) {
		for(int c=random2; c < random2+16; c++) {
			result.at<Vec3b>(r,c)[0] = patch.at<Vec3b>(i,j)[0]; 	//blue
			result.at<Vec3b>(r,c)[1] = patch.at<Vec3b>(i,j)[1];	//green
			result.at<Vec3b>(r,c)[2] = patch.at<Vec3b>(i,j)[2];	//red	
			j++;	
		}
		i++;
	}
	
}

void drawRandomRectanglesAndEllipses(Mat& img) {
	for(int i=0; i < 10; i++) {
		unsigned int random_size_r = std::rand()%100+10;
		unsigned int random_size_c = std::rand()%100+10;
		unsigned int random_size_ellipse;
		unsigned int random1 = std::rand()%img.rows;
		unsigned int random2 = std::rand()%img.cols;
		unsigned int random3, random4;
		if(random1 + random_size_r < img.rows) {
			random3 = random1+random_size_r;
		}
		else {
			random3 = img.rows;
		}
		if(random2 + random_size_c < img.cols) {
			random4 = random2+random_size_c;
		}
		else {
			random4 = img.cols;
		}
		
		rectangle(
			img,								// drawn in img
			Point(random2, random1),					// opposite vertices 
			Point(random4, random3),		
			Scalar(255,255,255),						// color white
			2,								// thickness
			8,								// lineType
			0);								// shift

		ellipse( img,
           		Point( random2, random1 ),
           		Size( random_size_c, random_size_r ),
			0,								// angle
			0,
			360,
			Scalar( 255, 255, 255),						// color white
			2,								// thickness
			8 );								// lineType
	}
}

int main(int argc, char* argv[])
{
    // check input arguments
    if(argc!=2) {
        cout << "usage: " << argv[0] << " <path_to_image>" << endl;
        return -1;
    }

    // read and display the input image
    Mat bgrImg;
    readImage(argv[1], bgrImg);
    display("(a) inputImage", bgrImg);

    // (b) convert bgrImg to grayImg
    Mat grayImg;
    convertToGrayImg(bgrImg, grayImg);
    display("(b) grayImage", grayImg);

    // (c) bgrImg - 0.5*grayImg
    Mat imgC;
    subtractIntensityImage(bgrImg, grayImg, imgC);
    display("(c) subtractedImage", imgC);

    // (d) pixelwise operations
    Mat imgD;
    pixelwiseSubtraction(bgrImg, grayImg, imgD);
    display("(d) subtractedImagePixelwise", imgD);

    // (e) copy 16x16 patch from center to a random location
    Mat patch;
    extractPatch(bgrImg, patch);
    display("(e) random patch", patch);
    Mat imgE;
    copyPatchToRandomLocation(bgrImg, patch, imgE);
    display("(e) afterRandomCopying", imgE);

    // (f) drawing random rectanges on the image
    drawRandomRectanglesAndEllipses(bgrImg);
    display("(f) random elements", bgrImg);

    return 0;
}
