#include <iostream>

#include <opencv2/opencv.hpp>


void customHoughCircles(  const cv::Mat                &ImageGRAY,
                                std::vector<cv::Vec3f> &detectedCircles,      // OUT
                          const int              numberOfBestCirclesReturned, // 1 - numberOfBestCirclesReturned (sorted)
                          const int              accumDivider,                // 2 - accumulator resolution (sizeImage/XXX)
                          const double           cannyHighThresh,             // 3 - Canny high threshold (low = 50%)
                          const double           minRadius,                   // 4 - min radius
                          const double           maxRadius,                   // 5 - max radius
                          const int              radiiNumb,                   // 6 - number of radii to check
                          const double           thetaStep_Degrees            // 7 - resolution of theta (degrees!)
                       );

void part1_1();
void part1_2();
void part1_3();
void part2_1();
void part2_2();
void part2_3();
void part3();

std::string  PATH_Circles = "./images/circles.png";
std::string  PATH_Face    = "./images/face2.png";
std::string  PATH_Flower  = "./images/flower.png";


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main()
{

    // Uncomment the part of the exercise that you wish to implement.
    // For the final submission all implemented parts should be uncommented.


    part2_1();
    part2_2();
    part2_3();
    part3();
    part1_1();
    part1_2();
    part1_3();

    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    END    /////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void draw_circles(std::vector<cv::Vec3f> circles, cv::Mat& im) {
    for (size_t i = 0; i < circles.size(); i++) {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = circles[i][2];

        cv::circle(im, center, radius, cv::Scalar(0, 255, 0), 3, 8, 0);
    }
}

void part1_1()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1 - 1    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    cv::Mat       im_Circles_BGR = cv::imread( PATH_Circles );
    // BGR to Gray
    cv::Mat                       im_Circles_Gray;
    cv::cvtColor( im_Circles_BGR, im_Circles_Gray, cv::COLOR_BGR2GRAY );

    // Synthetic image - No Blurring necessary for denoising !!!
    // But it only detects circles after blurring...
    cv::GaussianBlur(im_Circles_Gray, im_Circles_Gray, cv::Size(9,9), 2, 2);

    int canny_upper = 50;
    int canny_lower = 10;
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(im_Circles_Gray, circles, cv::HOUGH_GRADIENT, 1, 1, canny_upper, canny_lower, 5, 5);

    draw_circles(circles, im_Circles_BGR);

    cv::namedWindow("Part 1.1", cv::WINDOW_AUTOSIZE);
    cv::imshow("Part 1.1", im_Circles_BGR);
    cv::waitKey(0);

    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void detect_circles(cv::Mat& im, cv::Mat& accumulator, std::vector<cv::Vec3f>& circles, int radius, double threshold) {
    cv::Mat edges;
    cv::Mat gradients_x;
    cv::Mat gradients_y;
    cv::Mat magnitude;
    cv::Mat angles;
    accumulator = cv::Mat::zeros(im.size(), CV_8UC1);

    // Extract edges and gradients
    cv::Canny(im, edges, 50, 10);
    cv::Sobel(im, gradients_x, CV_32F, 1, 0, 3);
    cv::Sobel(im, gradients_y, CV_32F, 0, 1, 3);
    cv::cartToPolar(gradients_x, gradients_y, magnitude, angles, false);

    // Fill accumulator
    for (int i = 0; i < im.rows; i++) {
        for (int j = 0; j < im.cols; j++) {
            if (edges.at<uchar>(i, j) > 0) {
                for (int d = 0; d < 360; d++) {
                    double a = i - radius * cos(d);
                    double b = j - radius * sin(d);
                    if (a < im.cols && b < im.rows) accumulator.at<uchar>(round(a), round(b)) += 1;
                }
            }
        }
    }
    cv::normalize(accumulator, accumulator, 0.0, 255.0, cv::NORM_MINMAX);

    // Find circles
    for (int i = 0; i < accumulator.rows; i++) {
        for (int j = 0; j < accumulator.cols; j++) {
            if (accumulator.at<uchar>(i, j) > threshold) {
                circles.push_back(cv::Vec3f(j, i, radius));
            }
        }
    }
}


void part1_2()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1 - 2    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    cv::Mat im_Circles_BGR = cv::imread( PATH_Circles );
    // BGR to Gray
    cv::Mat                       im_Circles_Gray;
    cv::cvtColor( im_Circles_BGR, im_Circles_Gray, cv::COLOR_BGR2GRAY );
    // Synthetic image - No Blurring necessary for denoising !!!
    // But it only detects circles after blurring...
    //cv::GaussianBlur(im_Circles_Gray, im_Circles_Gray, cv::Size(9,9), 2, 2);

    // Perform the steps described in the exercise sheet
    cv::Mat accumulator;
    std::vector<cv::Vec3f> circles;

    detect_circles(im_Circles_Gray, accumulator, circles, 5, 220);
    draw_circles(circles, im_Circles_BGR);

    cv::namedWindow("Part 1.2", cv::WINDOW_AUTOSIZE);
    cv::imshow("Part 1.2", im_Circles_BGR);

    cv::namedWindow("Accumulator", cv::WINDOW_AUTOSIZE);
    cv::imshow("Accumulator", accumulator);
    cv::waitKey(0);

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    // If needed perform normalization of the image to be displayed

    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part1_3()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1 - 3    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    std::cout << std::endl << "Please Wait... CustomHoughCircles @ face..." << std::endl;

    cv::Mat im_Face_BGR = cv::imread( PATH_Face );
    cv::Mat im_Face_Gray;

    // BGR to Gray
    cv::cvtColor( im_Face_BGR, im_Face_Gray, cv::COLOR_BGR2GRAY );
    // Blur for denoising
    cv::GaussianBlur( im_Face_Gray, im_Face_Gray, cv::Size(0,0), 1,3 );
    // Erosion/Dilation to eliminate some noise
    int erDil = 1;
    //cv::erode(  im_Face_Gray, im_Face_Gray, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(erDil,erDil) ) );
    //cv::dilate( im_Face_Gray, im_Face_Gray, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(erDil,erDil) ) );

    // Perform the steps described in the exercise sheet
    cv::Mat accumulator;
    std::vector<cv::Vec3f> circles;

    detect_circles(im_Face_Gray, accumulator, circles, 4, 220);
    draw_circles(circles, im_Face_BGR);

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    // If needed perform normalization of the image to be displayed*cv::namedWindow("Part 1.3", cv::WINDOW_AUTOSIZE);
    cv::imshow("Part 1.3", im_Face_BGR);

    cv::namedWindow("Accumulator", cv::WINDOW_AUTOSIZE);
    cv::imshow("Accumulator", accumulator);
    cv::waitKey(0);
    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part2_1()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2 - 1    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    // read the image file flower
    cv::Mat                                              flower;
    cv::imread( PATH_Flower, cv::IMREAD_COLOR).convertTo(flower, CV_32FC3, (1./255.)); // image normalized [0,255] -> [0,1]

    // gray version of flower
    cv::Mat              flower_gray;
    cv::cvtColor(flower, flower_gray, CV_BGR2GRAY);
    cv::imshow("original image",                         flower_gray);
    // Perform the steps described in the exercise sheet
    cv::Mat samples = flower_gray.reshape(1, 1).t();
    cv::Mat labels;
    cv::Mat centers;
    cv::Mat results(flower_gray.size(), flower_gray.type());
    //cv::kmeans(samples, 10, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, 0.0001), 1, cv::KMEANS_RANDOM_CENTERS, centers);
    cv::kmeans(samples, 10, labels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), 5, cv::KMEANS_PP_CENTERS, centers);
    for ( int j = 0; j < flower_gray.rows; j++ ) {
        for ( int i = 0; i < flower_gray.cols; i++ ) {
            int cluster_label = labels.at<int>(i + j*flower_gray.cols, 0);
            results.at<uchar>(i, j) = centers.at<uchar>(cluster_label);
        }
    }

    //cv::normalize(results, results, 255.0, 0.0, cv::NORM_MINMAX);
    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
    cv::imshow("result", results);

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**

    cv::waitKey(0);
    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part2_2()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2 - 2    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    std::cout << "\n" << "kmeans with color" << std::endl;

    // read the image file flower
    cv::Mat                                              flower;
    cv::imread( PATH_Flower, cv::IMREAD_COLOR).convertTo(flower, CV_32FC3, (1./255.)); // image normalized [0,255] -> [0,1]
    // gray version of flower
    cv::Mat              flower_gray;
    cv::cvtColor(flower, flower_gray, CV_BGR2GRAY);
    cv::imshow("original image", flower);

    // Perform the steps described in the exercise sheet
    cv::Mat samples = flower_gray.reshape(1, flower.rows * flower.cols).t();
    /*cv::Mat samples(flower.rows * flower.cols, 3, CV_32F);
    for (int j = 0; j < flower.rows; j++) {
        for (int i = 0; j < flower.cols; j++) {
            for ( int k = 0; k < 3; k++) {
                samples.at<float>(j + i*flower.rows, k) = flower.at<cv::Vec3b>(j, i)[k];
            }
        }
    }*/

    cv::Mat labels;
    cv::Mat centers;
    //cv::kmeans(samples, 10, labels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), 5, cv::KMEANS_PP_CENTERS, centers);
    cv::kmeans(samples, 10, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, 0.0001), 1, cv::KMEANS_RANDOM_CENTERS, centers);
    std::cout << "Size labels: " << labels.size() << std::endl;
    cv::Mat result(flower.size(), flower.type());
    for (int j = 0; j < flower.rows; j++) {
        for (int i = 0; i < flower.cols; i++) {
            int cluster_label = labels.at<int>(0, i + j*flower.cols);
            result.at<cv::Vec3b>(j, i)[0] = centers.at<float>(cluster_label, 0);
            result.at<cv::Vec3b>(j, i)[1] = centers.at<float>(cluster_label, 1);
            result.at<cv::Vec3b>(j, i)[2] = centers.at<float>(cluster_label, 2);
        }
    }

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
    cv::imshow("result", result);

    cv::waitKey(0);
    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part2_3()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2 - 3    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    std::cout << "\n" << "kmeans with gray and pixel coordinates" << std::endl;

    // read the image file flower
    cv::Mat                                              flower;
    cv::imread( PATH_Flower, cv::IMREAD_COLOR).convertTo(flower, CV_32FC3, (1./255.)); // image normalized [0,255] -> [0,1]
    // gray version of flower
    cv::Mat              flower_gray;
    cv::cvtColor(flower, flower_gray, CV_BGR2GRAY);
    cv::imshow("original image", flower);

    // Perform the steps described in the exercise sheet

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**

    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part3()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 3    //////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    // read the image file flower
    cv::Mat                                              flower;
    cv::imread( PATH_Flower, cv::IMREAD_COLOR).convertTo(flower, CV_32FC3, (1./255.)); // image normalized [0,255] -> [0,1]
    cv::imshow("original image",                         flower);
    // BGR -> LUV
    cv::Mat              flower_luv;
    cv::cvtColor(flower, flower_luv, CV_BGR2Luv);
    cv::waitKey(0);

    //current mass center
    double m00 = 0.;
    double m01 = 0.;
    double m10 = 0.;
    for(int row = 0; row < flower_luv.rows; row++) {
        for(int col = 0; col < flower_luv.cols; col++) {
            //flower_luv.at<cv::Vec3b>(row, col)[0];
            //std::cout << flower_luv.at<cv::Vec3b>(row, col)[0] << std::endl;
            m00 += flower_luv.at<cv::Vec3b>(row,col)[2];
            m10 += row * flower_luv.at<cv::Vec3b>(row,col)[0];
            m01 += col * flower_luv.at<cv::Vec3b>(row,col)[1];
            //double center_x = m01/m00;
            //double center_y = m10/m00;
            //std::cout << center_x << " ; " << center_y << std::endl;
        }
    }
    // sphere to mass center
    double center_x = m01/m00;
    double center_y = m10/m00;
    std::cout << center_x << " ; " << center_y << std::endl;
    std::cout << flower_luv.rows << " ; " << flower_luv.cols << std::endl;
    
    //for(int iterations = 0; iterations < 50;) {
        //cv::TermCriteria criteria;
        //cv::Rect window;
        //cv::Mat d = flower_luv.clone();
    //int iteratinos = cv::meanShift(d,window,(cv::TermCriteria) criteria);
    //cv::imshow("window", d);
    //}

    // Perform the steps described in the exercise sheet

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**

    //cv::namedWindow("Part 3: Traffic", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Part 3: Traffic", im_Traffic_Gray);



    cv::waitKey(0);

    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

