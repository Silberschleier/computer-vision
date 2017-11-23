#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#define PI 3.14159
std::string PATH_Image   = "./images/gnome.png";
cv::Rect bb_Image(92,65,105,296);

void part1__1(const cv::Mat& img, const cv::Mat& mask_fg, const cv::Mat& mask_bg);
void part1__2(const cv::Mat& img, const cv::Mat& mask_fg, const cv::Mat& mask_bg);

////////////////////////////////////
// class declaration for task 1_1 //
////////////////////////////////////

class GMM_opencv{
private:
    int num_clus;
    cv::Mat_<double> samples;               // add more variables if necessary
public:
    GMM_opencv();
    ~GMM_opencv();
    void init(const int nmix, const cv::Mat& img, const cv::Mat& mask);
    cv::Mat return_posterior(const cv::Mat& img);
};


////////////////////////////////////
// class declaration for task 1_2 //
////////////////////////////////////

class GMM_custom{
private:
    int num_clus;
    std::vector<double> wt;             // cache for E step + final model
    std::vector<cv::Mat_<double> > mu;
    std::vector<cv::Mat_<double> > cov;
    cv::Mat_<double> samples;           // training pixel samples
    cv::Mat_<double> posterior;         // posterior probability for M step
    int maxIter;

    bool performEM();                   // iteratively called by learnGMM()
public:
    GMM_custom();
    ~GMM_custom();
    void init(const int nmix, const cv::Mat& img, const cv::Mat& mask); // call this once per image
    void learnGMM();    // call this to learn GMM
    cv::Mat return_posterior(const cv::Mat& img);     // call this to generate probability map
};


////////////////////////////////////
// 2_* and 3 are theoretical work //
////////////////////////////////////

int main()
{

    // Uncomment the part of the exercise that you wish to implement.
    // For the final submission all implemented parts should be uncommented.
    cv::Mat img=cv::imread(PATH_Image);
    assert(img.rows*img.cols>0);
    cv::Mat mask_fg(img.rows,img.cols,CV_8U); mask_fg.setTo(0); mask_fg(bb_Image).setTo(255);
    cv::Mat mask_bg(img.rows,img.cols,CV_8U); mask_bg.setTo(255); mask_bg(bb_Image).setTo(0);
    cv::Mat show=img.clone();
    cv::rectangle(show,bb_Image,cv::Scalar(0,0,255),1);
    cv::imshow("Image",show);
    cv::imshow("mask_fg",mask_fg);
    cv::imshow("mask_bg",mask_bg);
    cv::waitKey(0);

    part1__1(img,mask_fg,mask_bg);
    part1__2(img,mask_fg,mask_bg);

    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    END    /////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

}


void part1__1(const cv::Mat& img, const cv::Mat& mask_fg, const cv::Mat& mask_bg)
{
    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1__1  /////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

    /**
    int nmix=

    GMM_opencv gmm_fg;
    gmm_fg.init(nmix,img,mask_fg);
    gmm_fg.learnGMM();
    cv::Mat fg=gmm_fg.return_posterior(img);

    GMM_opencv gmm_bg;
    gmm_bg.init(nmix,img,mask_bg);
    gmm_bg.learnGMM();
    cv::Mat bg=gmm_bg.return_posterior(img);

    cv::Mat show=bg+fg;
    cv::divide(fg,show,show);
    show.convertTo(show,CV_8U,255);
    cv::imshow("gmm_opencv",show);
    cv::waitKey(0);
    **/

    cv::destroyAllWindows();
}


void part1__2(const cv::Mat& img, const cv::Mat& mask_fg, const cv::Mat& mask_bg)
{
    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1__2 //////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

    /**
    int nmix=

    GMM_custom gmm_fg;
    gmm_fg.init(nmix,img,mask_fg);
    gmm_fg.learnGMM();
    cv::Mat fg=gmm_fg.return_posterior(img);

    GMM_custom gmm_bg;
    gmm_bg.init(nmix,img,mask_bg);
    gmm_bg.learnGMM();
    cv::Mat bg=gmm_bg.return_posterior(img);

    cv::Mat show=bg+fg;
    cv::divide(fg,show,show);
    show.convertTo(show,CV_8U,255);
    cv::imshow("gmm_custom",show);
    cv::waitKey(0);
    **/

    cv::destroyAllWindows();
}
