#include <iostream>

#include <opencv2/opencv.hpp>


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part1();
void part2();

std::string PATH_Ball   = "./images/ball.png";
std::string PATH_Coffee = "./images/coffee.png";


//////////////////////////////////////
// function declarations for task 1 //
//////////////////////////////////////
void  drawSnake(             cv::Mat  img, const std::vector<cv::Point2i>& snake);
void  snakes(          const cv::Mat& img, const cv::Point2i center, const int radius, std::vector<cv::Point2i>& snake);


//////////////////////////////////////
// function declarations for task 2 //
//////////////////////////////////////
void    showGray(          const cv::Mat& img, const std::string title="Image", const int t=0);
void    showContour(       const cv::Mat& img, const cv::Mat& contour,          const int t=0);
void    levelSetContours(  const cv::Mat& img, const cv::Point2f center,        const float radius, cv::Mat& phi);
cv::Mat computeContour(    const cv::Mat& phi, const float level );


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main()
{

    // Uncomment the part of the exercise that you wish to implement.
    // For the final submission all implemented parts should be uncommented.

    part1();
    part2();

    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    END    /////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part1()
{
    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1    //////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

    cv::Mat                                               ball;
    cv::imread( PATH_Ball  , cv::IMREAD_COLOR).convertTo( ball,   CV_32FC3, (1./255.) );
    cv::Mat                                               coffee;
    cv::imread( PATH_Coffee, cv::IMREAD_COLOR).convertTo( coffee, CV_32FC3, (1./255.) );

    std::vector<cv::Point2i>    snake;
    size_t                      radius;
    cv::Point2i                 center;

    std::cout << "ball image" << std::endl;
    // for snake initialization
    center = cv::Point2i( ball.cols/2, ball.rows/2 );
    radius = std::min(    ball.cols/3, ball.rows/3 );
    //////////////////////////////////////
    snakes( ball, center, radius, snake );
    //////////////////////////////////////

    std::cout << "coffee image" << std::endl;
    // for snake initialization
    center = cv::Point2i( coffee.cols/2, coffee.rows/2 );
    radius = std::min(    coffee.cols/3, coffee.rows/3 );
    ////////////////////////////////////////
    snakes( coffee, center, radius, snake );
    ////////////////////////////////////////

    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part2()
{
    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2    //////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

    cv::Mat                                               ball;
    cv::imread( PATH_Ball  , cv::IMREAD_COLOR).convertTo( ball,   CV_32FC3, (1./255.) );
    cv::Mat                                               coffee;
    cv::imread( PATH_Coffee, cv::IMREAD_COLOR).convertTo( coffee, CV_32FC3, (1./255.) );

    cv::Mat                     phi;
    size_t                      radius;
    cv::Point2i                 center;

    std::cout << "ball image" << std::endl;
    center = cv::Point2i( ball.cols/2, ball.rows/2 );
    radius = std::min(    ball.cols/3, ball.rows/3 );
    /////////////////////////////////////////////////////////
    levelSetContours(     ball,    center, radius, phi     );
    /////////////////////////////////////////////////////////

    std::cout << "coffee image" << std::endl;
    center = cv::Point2f( coffee.cols/2.f, coffee.rows/2.f );
    radius =    std::min( coffee.cols/3.f, coffee.rows/3.f );
    /////////////////////////////////////////////////////////
    levelSetContours(     coffee,  center, radius, phi     );
    /////////////////////////////////////////////////////////

    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////
// apply the snake algorithm to an image //
///////////////////////////////////////////
void snakes( const cv::Mat&                     img,
             const cv::Point2i                  center,
             const int                          radius,
                   std::vector<cv::Point2i>&    snake )
{
    // initialize snake with a circle
    const int     vvvTOTAL =  radius*CV_PI/7; // defines number of snake vertices // adaptive based on the circumference
    snake.resize( vvvTOTAL );
    float angle = 0;
    for (cv::Point2i& vvv: snake)
    {
        vvv.x = round( center.x + cos(angle)*radius );
        vvv.y = round( center.y + sin(angle)*radius );

        angle += 2*CV_PI/vvvTOTAL;
    }

    // visualization
    cv::Mat     vis;
    img.copyTo( vis );
    drawSnake(  vis, snake);
    ///////////////////////////////////////////////////////////
    std::cout << "Press any key to continue...\n" << std::endl;
    ///////////////////////////////////////////////////////////
    cv::imshow("Snake", vis);
    cv::waitKey();

    // Perform optimization of the initialized snake as described in the exercise sheet and the slides.
    // You might want to apply some GaussianBlur on the edges so that the snake sidles up better.
    // Iterate until
    // - optimal solution for every point is the center of a 3x3 (or similar) box, OR
    // - until maximum number of iterations is reached
for(int iteration = 0; iteration < 100; iteration++) {

//sheet02 - start
    cv::Mat img_BGR = img.clone();
    // Blur to denoise
    cv::GaussianBlur( img_BGR, img_BGR, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    // BGR to Gray
    cv::Mat                           img_Gray;
    cv::cvtColor(     img_BGR, img_Gray, cv::COLOR_BGR2GRAY ); // cv::COLOR_BGR2GRAY // CV_BGR2GRAY

    // Perform the computations asked in the exercise sheet
    cv::Mat gradients_x;
    cv::Mat gradients_y;
    cv::Sobel(img_Gray, gradients_x, CV_64F, 1, 0, 3);
    cv::Sobel(img_Gray, gradients_y, CV_64F, 0, 1, 3);
//sheet02 - end

    int states = 9;
    double energy[states][snake.size()];
    int position[states][snake.size()];
    // all zero 

    for(int i=0; i < (snake.size()-1); i++) {
        for(int j=0; j < states; j++) {
            energy[j][i] = 0.;
            position[j][i] = 0;
        }
    }
    for(int v=0; v < snake.size(); v++) {
        // go through all states of the current Vertex
        int states = 0;
        for(int x=-1; x <= 1; x++) {
        for(int y=-1; y <= 1; y++) {
            // unary U_n
            energy[states][v] = (-1)*( gradients_x.at<uchar>(snake[v].x+x, snake[v].y+y)*gradients_x.at<uchar>(snake[v].x+x, snake[v].y+y) + gradients_y.at<uchar>(snake[v].x+x, snake[v].y+y)*gradients_y.at<uchar>(snake[v].x+x, snake[v].y+y) );
        states++;
            //std::cout << "states: " << states << " ; v: " << v << " ; Energy: "<< energy[states][v] << std::endl;
        }
        }
        //std::cout << std::endl;
    }

    double alpha = 1;
    double min_energy;
    int min_states;
    for(int v=0; v < (snake.size()-1); v++) {
        // go through all states of the next Vertex
        int next_states = 0;
        for(int next_x=-1; next_x <= 1; next_x++) {
        for(int next_y=-1; next_y <= 1; next_y++) {
            // go through all states of the current Vertex
            int cur_states = 0;
            for(int cur_x=-1; cur_x <= 1; cur_x++) {
            for(int cur_y=-1; cur_y <= 1; cur_y++) {

                cv::Point2i nextVertex = snake[v+1];
                nextVertex.x += next_x;
                nextVertex.y += next_y;
                cv::Point2i curVertex = snake[v];
                curVertex.x += cur_x;
                curVertex.y += cur_y;
              
                double cur_energy = energy[cur_states][v] + alpha*((nextVertex.x-curVertex.x)*(nextVertex.x-curVertex.x)+(nextVertex.y-curVertex.y)*(nextVertex.y-curVertex.y)) + energy[next_states][v+1];
                if(cur_states == 0) {
                    min_energy = cur_energy;
                    min_states = cur_states;
                }
                else if(cur_energy < min_energy) {
                    min_energy = cur_energy;
                    min_states = cur_states;
                }

            cur_states++;
            }
            }
        energy[next_states][v+1] = min_energy;
        position[next_states][v+1] = min_states;

        next_states++;
        }
        }
    }


    // find min in the last column (the column of the last vertex)
    double min_last_energy;
    int last_states;
    for(int s=0; s < states; s++) {
        if(s == 0) {
            min_last_energy = energy[s][snake.size()-1];
            last_states = s;
            //std::cout << "s: " << s << std::endl;
        }
        else if(energy[s][snake.size()-1] < min_last_energy) {
            min_last_energy = energy[s][snake.size()-1];
            last_states = s;
            //std::cout << "s: " << s << " ; Energy: " << energy[s][snake.size()-1] << std::endl;
        }
    }

    // backtracking
    int final_states[snake.size()];
    final_states[snake.size()-1] = last_states;
    for(int s = (snake.size()-2); s >= 0; s--) {
        final_states[s] = position[final_states[s+1]][s];
    }


    for(int i=0; i < snake.size(); i++) {
        //std::cout << "snake " << i << " befor: " << snake[i] << std::endl;
        if(final_states[i] == 0) {
            snake[i].x -= 1;
            snake[i].y -= 1;
        }
        else if (final_states[i] == 1) {
            snake[i].x -= 1;
        }
        else if (final_states[i] == 2) {
            snake[i].x -= 1;
            snake[i].y += 1;
        }
        if(final_states[i] == 3) {
            snake[i].y -= 1;
        }
        else if (final_states[i] == 4) {
        }
        else if (final_states[i] == 5) {
            snake[i].y += 1;
        }
        if(final_states[i] == 6) {
            snake[i].x += 1;
            snake[i].y -= 1;
        }
        else if (final_states[i] == 7) {
            snake[i].x += 1;
        }
        else if (final_states[i] == 8) {
            snake[i].x += 1;
            snake[i].y += 1;
        }
        //std::cout << "snake " << i << " after: " << snake[i] << std::endl << std::endl;
    }

    // At each step visualize the current result
    // using **drawSnake() and cv::waitKey(10)** as in the example above and when necessary **std::cout**
    // In the end, after the last visualization, use **cv::destroyAllWindows()**

    // visualization
    cv::Mat     result;
    img.copyTo( result );
    drawSnake(  result, snake);
    cv::imshow("Snake result", result);
    cv::waitKey();
}
    cv::destroyAllWindows();
}


////////////////////////////////
// draws a snake on the image //
////////////////////////////////
void drawSnake(       cv::Mat                   img,
                const std::vector<cv::Point2i>& snake )
{
    const size_t siz = snake.size();

    for (size_t iii=0; iii<siz; iii++)
        cv::line( img, snake[iii], snake[(iii+1)%siz], cv::Scalar(0,0,1) );

    for (const cv::Point2i& p: snake)
        cv::circle( img, p, 2, cv::Scalar(1,0,0), -1 );
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void gradient_descent(cv::Mat& phi, cv::Mat& w, cv::Mat& w_grad_x, cv::Mat& w_grad_y) {
    double tau = 0.05;
    double epsilon = 0.001;
    int sobel_kernel_size = 1;

    //cv::Mat w(phi.size(), CV_32FC1);
    //cv::pow(phi, 2, phi2);

    cv::Mat phi_new(phi.size(), phi.type());

    cv::Mat grad_phi_x, grad_phi_y, grad_phi_xx, grad_phi_yy, grad_phi_xy;
    //cv::Mat grad_phi2_x, grad_phi2_y;
    /*cv::Sobel(phi, grad_phi_x, CV_32FC1, 1, 0, 1);
    cv::Sobel(phi, grad_phi_y, CV_32FC1, 0, 1, 1);
    cv::Sobel(phi, grad_phi_xx, CV_32FC1, 2, 0, 1);
    cv::Sobel(phi, grad_phi_yy, CV_32FC1, 0, 2, 1);
    cv::Sobel(phi, grad_phi_xy, CV_32FC1, 1, 1, 1);*/
    //cv::Sobel(phi2, grad_phi2_x, CV_32FC1, 1, 0, sobel_kernel_size);
    //cv::Sobel(phi2, grad_phi2_y, CV_32FC1, 0, 1, sobel_kernel_size);

    double mean_curv_motion, propagation, deriv_x, deriv_y, deriv_xx, deriv_yy, deriv_xy;
    for (int x=0; x<phi.rows; x++) {
        for (int y=0; y<phi.cols; y++) {
            mean_curv_motion = 0;
            propagation = 0;

            /*deriv_x = grad_phi_x.at<float>(x, y);
            deriv_y = grad_phi_y.at<float>(x, y);
            deriv_xx = grad_phi_xx.at<float>(x, y);
            deriv_yy = grad_phi_yy.at<float>(x, y);
            deriv_xy = grad_phi_xy.at<float>(x, y);*/

            deriv_x = 0.5 * (phi.at<float>(x+1, y) - phi.at<float>(x-1, y));
            deriv_y = 0.5 * (phi.at<float>(x, y+1) - phi.at<float>(x, y-1));

            if (std::sqrt(deriv_x * deriv_x + deriv_y * deriv_y) == 0) {
                deriv_xx = (1 / 2) * (phi.at<float>(x+1, y) - 2 * phi.at<float>(x, y) + phi.at<float>(x-1, y));
                deriv_yy = (1 / 2) * (phi.at<float>(x, y+1) - 2 * phi.at<float>(x, y) + phi.at<float>(x, y-1));
                deriv_xy = (1 / 4) * (phi.at<float>(x+1, y+1) - phi.at<float>(x+1, y-1) - phi.at<float>(x-1, y+1) + phi.at<float>(x-1, y-1));

                mean_curv_motion += deriv_xx * deriv_y * deriv_y;
                mean_curv_motion -= 2 * deriv_x * deriv_y * deriv_xy;
                mean_curv_motion += deriv_yy * deriv_x * deriv_x;
                mean_curv_motion /= deriv_x * deriv_x + deriv_y * deriv_y + epsilon;
                mean_curv_motion *= tau * w.at<float>(x, y);
            }

            double loc_w = tau * w_grad_x.at<float>(x, y);
            if (loc_w < 0)  propagation += loc_w * (phi.at<float>(x+1, y) - phi.at<float>(x, y));
            else            propagation += loc_w * (phi.at<float>(x, y) - phi.at<float>(x-1, y));

            loc_w = tau * w_grad_y.at<float>(x, y);
            if (loc_w < 0)  propagation += loc_w * (phi.at<float>(x, y+1) - phi.at<float>(x, y));
            else            propagation += loc_w * (phi.at<float>(x, y) - phi.at<float>(x, y-1));

            phi_new.at<float>(x, y) = phi.at<float>(x, y) + mean_curv_motion + propagation;
        }
    }
    phi_new.copyTo(phi);
}

void calculate_w(cv::Mat& img, cv::Mat& w) {
    cv::Mat img_grad_x, img_grad_y;
    cv::Mat img_magnitudes, img_angles;

    cv::Sobel(img, img_grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(img, img_grad_y, CV_32F, 0, 1, 3);
    cv::cartToPolar(img_grad_x, img_grad_y, img_magnitudes, img_angles, false);

    cv::pow(img_magnitudes, 2, img_magnitudes);

    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            w.at<float>(x, y) = 1 / (img_magnitudes.at<float>(x, y) + 1);
        }
    }
}

///////////////////////////////////////////////////////////
// runs the level-set geodesic active contours algorithm //
///////////////////////////////////////////////////////////
void levelSetContours( const cv::Mat& img, const cv::Point2f center, const float radius, cv::Mat& phi )
{
    phi.create( img.size(), CV_32FC1 );
    //////////////////////////////
    // signed distance map **phi**
    //////////////////////////////
    // initialize as a cone around the
    // center with phi(x,y)=0 at the radius
    for     (int y=0; y<phi.rows; y++)   {   const  float         disty2 = pow( y-center.y, 2 );
        for (int x=0; x<phi.cols; x++)       phi.at<float>(y,x) = disty2 + pow( x-center.x, 2 );   }
                              cv::sqrt( phi, phi );

    // positive values inside
    phi = (radius - phi);
    cv::Mat temp = computeContour( phi, 0.0f);

    ///////////////////////////////////////////////////////////
    std::cout << "Press any key to continue...\n" << std::endl;
    ///////////////////////////////////////////////////////////
    showGray(    phi, "phi", 1 );
    showContour( img, temp,  0 );
    /////////////////////////////

    // Perform optimization of the initialized level-set function with geodesic active contours as described in the exercise sheet and the slides.
    // Iterate until
    // - the contour does not change between 2 consequitive iterations, or
    // - until a maximum number of iterations is reached

    cv::Mat img_gray;
    cv::cvtColor( img, img_gray, cv::COLOR_BGR2GRAY );

    cv::Mat w(phi.size(), CV_32FC1);
    cv::Mat w_grad_x, w_grad_y;

    for (int k = 0; k < 17000; k++) {
        calculate_w(img_gray, w);
        cv::Sobel(w, w_grad_x, CV_32FC1, 1, 0, 1);
        cv::Sobel(w, w_grad_y, CV_32FC1, 0, 1, 1);

        gradient_descent(phi, w, w_grad_x, w_grad_y);

        if (k % 100 == 0) {
            showGray(    phi, "phi", 1 );
            temp = computeContour( phi, 0.0f);
            showContour( img, temp,  1 );
            std::cout << "k = "<< k << std::endl;
        }
    }

    // At each step visualize the current result
    // using **showGray() and showContour()** as in the example above and when necessary **std::cout**
    // In the end, after the last visualization, use **cv::destroyAllWindows()**

    cv::destroyAllWindows();
}


////////////////////////////
// show a grayscale image //
////////////////////////////
void showGray( const cv::Mat& img, const std::string title, const int t )
{
    CV_Assert( img.channels() == 1 );

    double               minVal,  maxVal;
    cv::minMaxLoc( img, &minVal, &maxVal );

    cv::Mat            temp;
    img.convertTo(     temp, CV_32F, 1./(maxVal-minVal), -minVal/(maxVal-minVal));
    cv::imshow( title, temp);
    cv::waitKey(t);
}


//////////////////////////////////////////////
// compute the pixels where phi(x,y)==level //
//////////////////////////////////////////////
cv::Mat computeContour( const cv::Mat& phi, const float level )
{
    CV_Assert( phi.type() == CV_32FC1 );

    cv::Mat segmented_NORMAL( phi.size(), phi.type() );
    cv::Mat segmented_ERODED( phi.size(), phi.type() );

    cv::threshold( phi, segmented_NORMAL, level, 1.0, cv::THRESH_BINARY );
    cv::erode(          segmented_NORMAL, segmented_ERODED, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size2i(3,3)) );

    return ( segmented_NORMAL != segmented_ERODED );
}


///////////////////////////
// draw contour on image //
///////////////////////////
void showContour( const cv::Mat& img, const cv::Mat& contour, const int t )
{
    CV_Assert( img.cols == contour.cols   &&
               img.rows == contour.rows   &&
               img.type()     == CV_32FC3 &&
               contour.type() == CV_8UC1  );

    cv::Mat temp( img.size(), img.type() );

    const cv::Vec3f color( 0, 0, 1 ); // BGR

    for     (int y=0; y<img.rows; y++)
        for (int x=0; x<img.cols; x++)
            temp.at<cv::Vec3f>(y,x) = contour.at<uchar>(y,x)!=255 ? img.at<cv::Vec3f>(y,x) : color;

    cv::imshow("contour", temp);
    cv::waitKey(t);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

