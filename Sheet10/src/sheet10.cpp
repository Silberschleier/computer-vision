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
    cv::Size chessboard_dimensions(10, 7);

    for(int i=1; i<NUM_IMAGES; i++){
        string fname=fixedLenString(i,3,image_prefix,image_suffix);
        cv::Mat img=cv::imread(fname.c_str());
        cv::imshow("frame",img);

        // task 1: call function

        cv::Mat corners;
        auto patternWasFound = cv::findChessboardCorners(img, chessboard_dimensions, corners);
        // TODO: Maybe refine with cornerSubPix
        cv::drawChessboardCorners(img, chessboard_dimensions, corners, patternWasFound);
        cv::imshow("pattern", img);

        cv::waitKey(0);
    }

	// task 2: call function

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
