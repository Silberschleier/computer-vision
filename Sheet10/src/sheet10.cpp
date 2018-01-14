#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iomanip>
#include <cv.h>

using namespace std;

const int NUM_IMAGES=14;
string image_prefix = "../images/frame";
string image_suffix = ".png";

//////////////////////////////////////
//////// utility functions ///////////
//////////////////////////////////////
string fixedLenString(int i,int len, string prefix, string suffix){
    stringstream ss;
    ss << setw(len) << setfill('0') << i;
    string s = ss.str();
    return prefix+s+suffix;
}

template < class T >
ostream& operator << (ostream& os, const vector<T>& v)
{
    os << "[";
    for (typename vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii)
    {
        os << " " << *ii;
    }
    os << "]";
    return os;
}

int main()
{
    cv::Size chessboard_dimensions(10, 7), img_size;
    std::vector<cv::Mat> images_color, images_gray, corners;

    for(int i=1; i<NUM_IMAGES; i++){
        string fname=fixedLenString(i,3,image_prefix,image_suffix);
        cv::Mat img=cv::imread(fname.c_str());
        img_size = img.size();
        images_color.push_back(img);
        cv::imshow("frame",img);

        // task 1: call function

        cv::Mat img_corners, img_gray;
        auto patternWasFound = cv::findChessboardCorners(img, chessboard_dimensions, img_corners);

        // Refine corners
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);
        cv::cornerSubPix(img_gray, img_corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));
        corners.push_back(img_corners);
        images_gray.push_back(img_gray);

        // Show corners
        cv::drawChessboardCorners(img, chessboard_dimensions, img_corners, patternWasFound);
        cv::imshow("pattern", img);

        cv::waitKey(0);
    }

    // task 2: call function
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distortionMatrix = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rotation_vectors, translation_vectors;
    double rms = cv::calibrateCamera(cv::Mat(), corners, img_size, cameraMatrix, distortionMatrix, rotation_vectors, translation_vectors);

	// task 3: call function

	// task 4: call function

	// task 5: call function

    cout <<                                                                                                   endl;
    cout << "////////////////////////////////////////////////////////////////////////////////////////////" << endl;
    cout << "////////////////////////////////////////////////////////////////////////////////////////////" << endl;
    cout << "////    END    /////////////////////////////////////////////////////////////////////////////" << endl;
    cout << "////////////////////////////////////////////////////////////////////////////////////////////" << endl;
    cout << "////////////////////////////////////////////////////////////////////////////////////////////" << endl;
    cout <<                                                                                                   endl;

}
