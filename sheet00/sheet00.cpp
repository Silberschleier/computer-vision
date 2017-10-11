#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void readImage(const char* file, Mat& result) {
    //TODO: implement your solution here
}

void display(const char* windowTitle, Mat& img) {
    //TODO: implement your solution here
}

void convertToGrayImg(Mat& img, Mat& result) {
    //TODO: implement your solution here
}

void subtractIntensityImage(Mat& bgrImg, Mat& grayImg, Mat& result) {
    //TODO: implement your solution here
}

void pixelwiseSubtraction(Mat& bgrImg, Mat& grayImg, Mat& result) {
    //TODO: implement your solution here
}

void extractPatch(Mat& img, Mat& result) {
    //TODO: implement your solution here
}

void copyPatchToRandomLocation(Mat& img, Mat& patch, Mat& result) {
    //TODO: implement your solution here
}

void drawRandomRectanglesAndEllipses(Mat& img) {
    //TODO: implement your solution here
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
