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
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<cv::Point3f> object_pattern;

    for ( int i=0; i < chessboard_dimensions.height; i++) {
        for ( int j=0; j < chessboard_dimensions.width; j++) {
            object_pattern.emplace_back(float(j), float(i), 0);
        }
    }

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

        objectPoints.push_back(object_pattern);

        cv::waitKey(100);
    }
    cv::waitKey(0);
    cv::destroyAllWindows();

    // task 2: call function
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distortionMatrix = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rotation_vectors, translation_vectors;
    double rms = cv::calibrateCamera(objectPoints, corners, img_size, cameraMatrix, distortionMatrix, rotation_vectors, translation_vectors);

    std::cout << "cameraMatrix" << std::endl;
    std::cout << "------------" << std::endl;
    std::cout << cameraMatrix << std::endl << std::endl;

    std::cout << "distortionMatrix" << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << distortionMatrix << std::endl;

    std::vector<std::vector<cv::Mat>> reprojection_table;
    for (int i = 0; i < rotation_vectors.size(); i++) {
        std::cout << std::endl << std::endl;
        cv::Mat rotation;
	
        cv::Rodrigues(rotation_vectors.at(i), rotation);
        std::cout << "Rotation for image " << i << ": " << std::endl << rotation << std::endl;
        std::cout << "Translation for image " << i << ": " << std::endl << translation_vectors.at(i) << std::endl;

	// task 3: call function
	cv::Mat translation = translation_vectors.at(i);
	// no need for the 3. col of the rotation matrix, because z=0
	cv::Mat rotation_and_translation(3,4,CV_64F);
	for(int row=0; row < 3; row++) {
		for(int col=0; col < 3; col++) {
			rotation_and_translation.at<double>(row,col) = rotation.at<double>(row,col);
		}
		rotation_and_translation.at<double>(row,3) = translation.at<double>(row,0);	
	}
	std::cout << "rotation and translation matrix " << std::endl << rotation_and_translation << std::endl;
	
	std::vector<cv::Point3f> o = objectPoints.at(i);
	std::vector<cv::Mat> reprojections;

	for(int points = 0; points < 70; points++ ) {
		cv::Point3f point = o.at(points);
		double vec[4] = {point.x, point.y, point.z, 1};
		cv::Mat position = cv::Mat(4,1, CV_64FC1, vec);
		cv::Mat reprojection = cameraMatrix * rotation_and_translation * position;
		reprojections.push_back(reprojection);
	}
	reprojection_table.push_back(reprojections);

	
	

        // task 4: call function
        cv::Mat undistorted, difference;
        cv::undistort(images_gray.at(i), undistorted, cameraMatrix, distortionMatrix);
        cv::absdiff(images_gray.at(i), undistorted, difference);

        cv::imshow("difference", difference);
        cv::waitKey(0);
    }
	// task 3: reprojection error
	std::cout << reprojection_table.size() << std::endl;
	double reprojection_error_x, reprojection_error_y = 0.;
	double n = reprojection_table.size();
	double k = reprojection_table.at(0).size();
	for(int i=0; i < reprojection_table.size(); i++) {
		std::vector<cv::Mat> reprojections = reprojection_table.at(i);
		std::vector<cv::Point3f> o = objectPoints.at(i);
		for(int j=0; j < reprojections.size(); j++) {
			cv::Mat pointR = reprojections.at(j);
			cv::Point3f pointO = o.at(j);
			reprojection_error_x += std::abs(pointR.at<double>(0,0)-pointO.x);
			reprojection_error_y += std::abs(pointR.at<double>(1,0)-pointO.y);
		}
	}
	reprojection_error_x = reprojection_error_x * (1/(n*k));
	reprojection_error_y = reprojection_error_y * (1/(n*k));
	std::cout << "reprojection_error_x = " << reprojection_error_x << std::endl;
	std::cout << "reprojection_error_y = " << reprojection_error_y << std::endl;

	// task 5: call function

    cout <<                                                                                                   endl;
    cout << "////////////////////////////////////////////////////////////////////////////////////////////" << endl;
    cout << "////////////////////////////////////////////////////////////////////////////////////////////" << endl;
    cout << "////    END    /////////////////////////////////////////////////////////////////////////////" << endl;
    cout << "////////////////////////////////////////////////////////////////////////////////////////////" << endl;
    cout << "////////////////////////////////////////////////////////////////////////////////////////////" << endl;
    cout <<                                                                                                   endl;

}
