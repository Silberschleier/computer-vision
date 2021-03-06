#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


void part1();
void part2();
void part3();
void part4();
void part5();
void buildMyPyramid( cv::Mat& src, std::vector<cv::Mat>& dst, int maxlevel); //part1 the gaussian pyramid
void buildLaplacianPyramid( cv::Mat& src, std::vector<cv::Mat>& dst, int maxlevel); //part1 the laplacian pyramid
void drawArrow(cv::Mat &image, cv::Point p, cv::Scalar color, double scaleMagnitude, double scaleArrowHead, double magnitube, double orientationDegrees);


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, char* argv[])
{

    // Uncomment the part of the exercise that you wish to implement.
    // For the final submission all implemented parts should be uncommented.

    part1();
    part2();
    part3();
    part4();
    part5();

    std::cout <<                                                            std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    END    //////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                            std::endl;

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part1()
{
    std::cout <<                                                            std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1    ///////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;

    // read the image file
    cv::Mat im_Traffic_BGR = cv::imread("./images/traffic.jpg", cv::IMREAD_COLOR);
    // gray version of bonn.png
    cv::Mat                      im_Traffic_Gray;
    cv::cvtColor(im_Traffic_BGR, im_Traffic_Gray, CV_BGR2GRAY);

    // construct Gaussian pyramids
    std::vector<cv::Mat>   gpyr;    // this will hold the Gaussian Pyramid created with OpenCV
    std::vector<cv::Mat> myGpyr;    // this will hold the Gaussian Pyramid created with your custom way

    // Please implement the pyramids as described in the exercise sheet, using the containers given above.

    //cv::buildPyramid (InputArray src, OutputArrayOfArrays dst, int maxlevel, int borderType=BORDER_DEFAULT) constructs the Gaussian pyramid for an image.
    int maxlevel = 5;
    cv::buildPyramid( im_Traffic_Gray, gpyr, maxlevel, cv::BORDER_DEFAULT);

    buildMyPyramid( im_Traffic_Gray, myGpyr, maxlevel);

    // Perform the computations asked in the exercise sheet and show them using **std::cout**
    // Compute the maximal pixel-wise difference between both versions for each layer.
    for(int i=0; i<= maxlevel; i++) {
        cv::Mat diff;
        cv::absdiff(gpyr[i], myGpyr[i], diff);
        double minVal, maxVal;
        minMaxLoc(diff, &minVal, &maxVal);
        std::cout << "maximum pixel error layer "<<  (char) (i+48) << ": " << maxVal << std::endl; //ASCII
    }

    // Show every layer of the pyramid
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    
    // Display the images of "gpyr"
    for(int i=0; i < gpyr.size(); i++) {
	char s[] = "gpyr layer  ";
	s[11] = (char) (i+48); //ASCII
        cv::namedWindow( s, cv::WINDOW_AUTOSIZE);
        cv::imshow(s, gpyr[i]);
        cv::waitKey(0);
    }
    cv::destroyAllWindows();
    // Display the images of "myGpyr"
    for(int i=0; i < myGpyr.size(); i++) {
	char s[] = "myGpyr layer   ";
	s[13] = (char) (i+48); //ASCII
        cv::namedWindow( s, cv::WINDOW_AUTOSIZE);
        cv::imshow(s, myGpyr[i]);
        cv::waitKey(0);
    }
    cv::destroyAllWindows();

    // For the laplacian pyramid you should define your own container.
    // If needed perform normalization of the image to be displayed

    // construct laplacian pyramids
    std::vector<cv::Mat>   lpyr;    // this will hold the laplacian Pyramid

    buildLaplacianPyramid( im_Traffic_Gray, lpyr, maxlevel);

    // Display the images of "lpyr"
    for(int i=0; i < lpyr.size(); i++) {
	char s[] = "lpyr layer   ";
	s[11] = (char) (i+48); //ASCII
        cv::namedWindow( s, cv::WINDOW_AUTOSIZE);
        cv::imshow(s, lpyr[i]);
        cv::waitKey(0);
    }

    cv::destroyAllWindows();
}

void buildMyPyramid( cv::Mat& src, std::vector<cv::Mat>& dst, int maxlevel) 
{
    cv::Mat gaussian_kernel = cv::getGaussianKernel(5, 2., CV_64F );
    
    dst.push_back(src.clone());
    for(int i=1; i <= maxlevel; i++) {
        
        dst.push_back(dst[i-1].clone());

	// use gaussian filter
        cv::filter2D(dst[i-1], dst[i], -1, gaussian_kernel);

	// remove every even-numbered row and column
        cv::Mat dst_i = dst[i](cv::Rect(0, 0, (int) (dst[i].cols/2.+0.5), (int) (dst[i].rows/2. + 0.5))); // size of dst[i] without even-numbered rows and columns
	int row_dst_i = 0;
        int col_dst_i = 0;
	for(int row=1; row < dst[i].rows; row=row+2) {
            col_dst_i = 0;
            for(int col=1; col < dst[i].cols; col=col+2) {
                dst_i.at<uchar>(row_dst_i, col_dst_i) = dst[i].at<uchar>(row,col);
                col_dst_i++;
            }
            row_dst_i++;
        }
	dst[i] = dst_i;
    }
}

void buildLaplacianPyramid( cv::Mat& src, std::vector<cv::Mat>& dst, int maxlevel) {
    // construct Gaussian pyramids
    std::vector<cv::Mat>   gpyr;    // this will hold the Gaussian Pyramid created with OpenCV
    cv::buildPyramid( src, gpyr, maxlevel, cv::BORDER_DEFAULT);

    // construct laplacian pyramids
    std::vector<cv::Mat>   lpyr(maxlevel+1);    // this will hold the laplacian Pyramid
    
    lpyr[maxlevel] = gpyr[gpyr.size()-1];
    for(int level=maxlevel-1; level>=0; level--) {
        // expand G_i+1
        cv::Mat expand;
        cv::pyrUp( gpyr[level+1], expand, cv::Size( gpyr[level].cols, gpyr[level].rows ) );

        // L_i = G_i - expand(G_i+1)
        cv::Mat laplacian = gpyr[level] - expand;
        lpyr[level] = laplacian;
    }
    dst = lpyr;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part2()
{
    std::cout <<                                                            std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2    ///////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;

    // apple and orange are CV_32FC3
    cv::Mat im_Apple_BGR, im_Orange_BGR, im_Apple, im_Orange;
    im_Apple = cv::imread("./images/apple.jpg",  cv::IMREAD_COLOR);
    im_Orange = cv::imread("./images/orange.jpg", cv::IMREAD_COLOR);
    //cv::cvtColor(im_Apple_BGR, im_Apple, CV_BGR2GRAY);
    //cv::cvtColor(im_Orange_BGR, im_Orange, CV_BGR2GRAY);
    //cv::imread("./images/apple.jpg",  cv::IMREAD_COLOR).convertTo(im_Apple,  CV_32FC3, (1./255.));
    //cv::imread("./images/orange.jpg", cv::IMREAD_COLOR).convertTo(im_Orange, CV_32FC3, (1./255.));
    cv::imshow("orange", im_Orange);
    cv::imshow("apple",  im_Apple );
    std::cout << "\n" << "Input images" << "   \t\t\t\t\t\t\t" << std::endl << "Press any key..." << std::endl;
    cv::waitKey(0);

    // Perform the blending using a Laplacian Pyramid

    // Build Laplacian pyramids LO and LA from images im_Orange and im_Apple
    int maxlevel = 7;
    std::vector<cv::Mat>   lpyr_LO;    // this will hold the laplacian Pyramid LO
    std::vector<cv::Mat>   lpyr_LA;    // this will hold the laplacian Pyramid LA
    cv::Mat im_O = im_Orange.clone();
    cv::Mat im_A = im_Apple.clone();
    buildLaplacianPyramid( im_O, lpyr_LO, maxlevel);
    buildLaplacianPyramid( im_A, lpyr_LA, maxlevel);

    //Build a Gaussian pyramid GR from selected region R

    cv::Mat im_R = im_Orange.clone();
    std::cout << "cols: " << im_Orange.cols << " rows: " << im_Orange.rows << std::endl;
    for(int row=0; row < im_Apple.rows; row++) {
        for(int col=0; col < im_Apple.cols/2; col++) {
            im_R.at<uchar>(row,col) = im_Apple.at<uchar>(row,col);
        }
    }
    std::vector<cv::Mat> lpyr_R = lpyr_LO;
    //buildLaplacianPyramid( im_R, lpyr_R, maxlevel);

    //buildLaplacianPyramid( im_R, lpyr_R, maxlevel);
    for(int i=0; i <= maxlevel; i ++) {
        for(int row=0; row < lpyr_LA[i].rows; row++) {
            for(int col=0; col < lpyr_LA[i].cols; col++) {
//                lpyr_R[i].at<uchar>(row,col) = im_R.at<uchar>(row,col)*lpyr_LA[i].at<uchar>(row,col)+(1-im_R.at<uchar>(row,col))*lpyr_LO[i].at<uchar>(row,col);

                if(col < im_Apple.cols/2) {
                    lpyr_R[i].at<uchar>(row,col) = lpyr_LA[i].at<uchar>(row,col);
                }

                else if(col > im_Apple.cols/2) {
                    lpyr_R[i].at<uchar>(row,col) = lpyr_LO[i].at<uchar>(row,col);
                }
                else {
                    lpyr_R[i].at<cv::Vec3b>(row,col)[0] =(int)( (lpyr_LA[i].at<cv::Vec3b>(row,col)[0] + lpyr_LO[i].at<cv::Vec3b>(row,col)[0])/2);
                }

            }
        }
    }


    //reconstruct gaussian pyramid
    std::vector<cv::Mat>   gpyr_R(maxlevel+1);    // this will hold the Gaussian Pyramid
    gpyr_R[maxlevel] = lpyr_R[maxlevel];

    for(int level = maxlevel-1; level >= 0; level--) {
        cv::Mat expand;
        cv::pyrUp( gpyr_R[level+1], expand, cv::Size( lpyr_R[level].cols, lpyr_R[level].rows ) );

        gpyr_R[level] = expand + 2*lpyr_R[level];
    }

    //cv::Mat result = gpyr_R[0];
    //result = result + 5* lpyr_R[0];
    //cv::imshow("result", result);
    //cv::waitKey(0);

    // Show the blending results @ several layers
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**

    // Display the images of "gpyr_R"
    for(int i=0; i < gpyr_R.size(); i++) {
	char s[] = "gpyr_R layer   ";
	s[13] = (char) (i+48); //ASCII
        cv::namedWindow( s, cv::WINDOW_AUTOSIZE);
        cv::imshow(s, gpyr_R[i]);
        cv::waitKey(0);
    }
    cv::destroyAllWindows();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part3()
{
    std::cout <<                                                            std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 3    ///////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;

    cv::Mat           im_Traffic_BGR = cv::imread("./images/traffic.jpg"); // traffic.jpg // circles.png
    // Blur to denoise
    cv::GaussianBlur( im_Traffic_BGR, im_Traffic_BGR, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    // BGR to Gray
    cv::Mat                           im_Traffic_Gray;
    cv::cvtColor(     im_Traffic_BGR, im_Traffic_Gray, cv::COLOR_BGR2GRAY ); // cv::COLOR_BGR2GRAY // CV_BGR2GRAY

    // Perform the computations asked in the exercise sheet
    cv::Mat gradients_x;
    cv::Mat gradients_y;
    cv::Sobel(im_Traffic_Gray, gradients_x, -1, 1, 0);
    cv::Sobel(im_Traffic_Gray, gradients_y, -1, 0, 1);
    int threshold = 250;

    cv::Mat arrows = cv::Mat::zeros(im_Traffic_Gray.size(), CV_8UC1);

    for (int i = 0; i < gradients_x.rows; i++) {
        for (int j = 0; j < gradients_x.cols; j += 10) {
            double grad_x = gradients_x.at<uchar>(i, j);
            double grad_y = gradients_y.at<uchar>(i, j);
            double magnitude = std::sqrt(pow(grad_x, 2) + pow(grad_y, 2));
            if (magnitude < threshold) continue;
            double angle = atan(grad_y / grad_x) * 180 / M_PI;
            drawArrow(arrows, cv::Point(i, j), cv::Scalar(254), 0.1, 10, magnitude, angle);
        }
    }

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**

    // Use the function **drawArrow** provided at the end of this file in order to
    // draw Vectors showing the Gradient Magnitude and Orientation
    // (to avoid clutter, draw every 10nth gradient,
    // only if the magnitude is above a threshold)

    cv::namedWindow("Part 3: Traffic", cv::WINDOW_AUTOSIZE);
    cv::imshow("Part 3: Traffic", im_Traffic_Gray);

    cv::namedWindow("Part 3: Arrows", cv::WINDOW_AUTOSIZE);
    cv::imshow("Part 3: Arrows", arrows);

    cv::waitKey(0);

    cv::destroyAllWindows();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part4()
{
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 4    ///////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;

    cv::Mat im_Traffic_BGR = cv::imread("./images/traffic.jpg");
    // BGR to Gray
    cv::Mat                       im_Traffic_Gray;
    cv::cvtColor( im_Traffic_BGR, im_Traffic_Gray, cv::COLOR_BGR2GRAY );
    cv::Mat edges;

    // Perform edge detection as described in the exercise sheet
    cv::Canny(im_Traffic_Gray, edges, 300, 100);

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    cv::namedWindow("Part 4: Traffic", cv::WINDOW_AUTOSIZE);
    cv::imshow("Part 4: Traffic", im_Traffic_Gray);
    cv::namedWindow("Part 4: Edges", cv::WINDOW_AUTOSIZE);
    cv::imshow("Part 4: Edges", edges);
    cv::waitKey(0);

    cv::destroyAllWindows();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double chamfer_distance(cv::Mat& im, cv::Mat& templ, int pos_x, int pos_y) {
    int count_edge_pixels = 0;
    double distance = 0;

    // Add all distances for each corresponding template edge pixel
    for (int i = 0; i < templ.rows; i++) {
        for (int j = 0; j < templ.cols; j++) {
            if (templ.at<uchar>(i, j) == 255) {
                if ( (pos_x + i < im.rows) && (pos_y + j < im.cols) ) {
                    distance += im.at<uchar>(pos_x + i, pos_y + j);
                } else distance += 255; // In case the specified position is outside of the image
                count_edge_pixels++;
            }
        }
    }
    return distance / count_edge_pixels;
}


void part5()
{
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 5    ///////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                            std::endl;

    // Read image, Blur to denoise
    cv::Mat                           im_Traffic_BGR = cv::imread("./images/traffic.jpg");
    cv::GaussianBlur( im_Traffic_BGR, im_Traffic_BGR,  cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    // BGR to Gray
    cv::Mat                           im_Traffic_Gray;
    cv::cvtColor(     im_Traffic_BGR, im_Traffic_Gray, cv::COLOR_BGR2GRAY );

    // Read Template
    cv::Mat im_Sign_BGR = cv::imread("./images/sign.png");
    cv::Mat im_Sign_Gray;
    // BGR to Gray
    cv::cvtColor( im_Sign_BGR, im_Sign_Gray, cv::COLOR_BGR2GRAY ); // cv::COLOR_BGR2GRAY // CV_BGR2GRAY
    cv::resize(im_Sign_Gray, im_Sign_Gray, cv::Size(70, 60));

    // Calculate distance transformation
    cv::Mat traffic_edges;
    cv::Canny(im_Traffic_Gray, traffic_edges, 500, 3);
    cv::Mat inverse_traffic_edges = cv::Scalar::all(255) - traffic_edges;
    cv::Mat transformation;
    cv::distanceTransform(inverse_traffic_edges, transformation, CV_DIST_L1, 3);
    cv::normalize(transformation, transformation, 0.0, 1.0, cv::NORM_MINMAX);

    // Extract edges of the sign
    cv::Mat sign_edges;
    cv::Canny(im_Sign_Gray, sign_edges, 350, 3);

    // Matching
    cv::Mat voting_space = cv::Mat::zeros(im_Traffic_Gray.size(), CV_8UC1);
    double distance;
    // Calculate chamfer distance for every position
    for (int i = 0; i < voting_space.rows; i++) {
        for (int j = 0; j < voting_space.cols; j++) {
            distance = chamfer_distance(transformation, sign_edges, i, j);
            voting_space.at<uchar>(i, j) = distance;
        }
    }

    double min, max;
    cv::Point min_loc, max_loc;

    // First sign
    cv::minMaxLoc(voting_space, &min, &max, &min_loc, &max_loc);
    cv::rectangle(im_Traffic_BGR, min_loc, cv::Point(min_loc.x + sign_edges.rows, min_loc.y + sign_edges.cols), cv::Scalar(255, 0, 0), 4);
    voting_space.at<uchar>(min_loc) = 255; // To 'forget' the first minimum

    // Second sign
    cv::minMaxLoc(voting_space, &min, &max, &min_loc, &max_loc);
    cv::rectangle(im_Traffic_BGR, min_loc, cv::Point(min_loc.x + sign_edges.rows, min_loc.y + sign_edges.cols), cv::Scalar(0, 0, 255), 4);

    //cv::normalize(voting_space, voting_space, 0.0, 1.0, cv::NORM_MINMAX);

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    // If needed perform normalization of the image to be displayed

    cv::namedWindow("Part 5: Distance Transformation", cv::WINDOW_AUTOSIZE);
    cv::imshow("Part 5: Distance Transformation", transformation);

    cv::namedWindow("Part 5: Edges Traffic", cv::WINDOW_AUTOSIZE);
    cv::imshow("Part 5: Edges Traffic", traffic_edges);

    cv::namedWindow("Part 5: Edges Sign", cv::WINDOW_AUTOSIZE);
    cv::imshow("Part 5: Edges Sign", sign_edges);

    cv::namedWindow("Part 5: Voting Space", cv::WINDOW_AUTOSIZE);
    cv::imshow("Part 5: Voting Space", voting_space);

    cv::namedWindow("Part 5: Result", cv::WINDOW_AUTOSIZE);
    cv::imshow("Part 5: Result", im_Traffic_BGR);
    cv::waitKey(0);

    cv::destroyAllWindows();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Use this function for visualizations in part3

void drawArrow(cv::Mat &image, cv::Point p, cv::Scalar color, double scaleMagnitude, double scaleArrowHead, double magnitube, double orientationDegrees)
{
    int arrowHeadAngleCoeff = 10;

    magnitube *= scaleMagnitude;

    double theta = orientationDegrees * M_PI / 180; // rad
    cv::Point q;
    q.x = p.x + magnitube * cos(theta);
    q.y = p.y + magnitube * sin(theta);

    //Draw the principle line
    cv::line(image, p, q, color);
    //compute the angle alpha
    double angle = atan2((double)p.y-q.y, (double)p.x-q.x);
    //compute the coordinates of the first segment
    p.x = (int) ( q.x + static_cast<int>(  round( scaleArrowHead * cos(angle + M_PI/arrowHeadAngleCoeff)) )  );
    p.y = (int) ( q.y + static_cast<int>(  round( scaleArrowHead * sin(angle + M_PI/arrowHeadAngleCoeff)) )  );
    //Draw the first segment
    cv::line(image, p, q, color);
    //compute the coordinates of the second segment
    p.x = (int) ( q.x + static_cast<int>(  round( scaleArrowHead * cos(angle - M_PI/arrowHeadAngleCoeff)) )  );
    p.y = (int) ( q.y + static_cast<int>(  round( scaleArrowHead * sin(angle - M_PI/arrowHeadAngleCoeff)) )  );
    //Draw the second segment
    cv::line(image, p, q, color);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
