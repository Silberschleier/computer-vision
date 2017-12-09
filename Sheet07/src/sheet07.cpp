#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define UNIT_VARIANCE 10.0

string file_for_procustes = "../images/hands_orig_train.txt";
string train_file_shape = "../images/hands_aligned_train.txt";
string test_file_shape = "../images/hands_aligned_test.txt";

/////////////////////////////////////////////////
//////// class declaration for task 1 ///////////
/////////////////////////////////////////////////
class ProcrustesAnalysis{
public:
    ProcrustesAnalysis();
    ~ProcrustesAnalysis(){};
    bool LoadData(string in_fpath);
    void AlignData();

private:
    void ComputeMeanShape(Mat& in_data, Mat& out_data, double in_unit_var);
    void AlignShape(Mat in_tgt, Mat& in_ref, Mat& out_aligned);
    float ComputeAvgError(Mat& in_newdata);
    void displayShape(Mat& shapes, string header, int waitFlag);
    cv::Mat data;
    int m_iter, m_maxIter;
    float m_err, m_maxErr;
    int num_coords, num_sampls;
};


ProcrustesAnalysis::ProcrustesAnalysis(){
    // set values ...
}


bool ProcrustesAnalysis::LoadData(string in_fpath){
    ifstream in(in_fpath.c_str());
    if(!in) return false;

    in >> num_coords;
    in >> num_sampls;
    Mat indata(num_coords,num_sampls,CV_32F);
    indata.setTo(0);
    for(int row=0; row<indata.rows; ++row){
        for(int col=0; col<indata.cols; ++col){
            in >> indata.at<float>(row,col);
        }
    }
    data = indata;
    return true;
}


void ProcrustesAnalysis::AlignData(){
    
    // Exercise 1.4: 
    // compute mean shape, align shapes to mean, compute error, repeat ...
    return;
}



void ProcrustesAnalysis::ComputeMeanShape(Mat& in_data, Mat& out_mu, double in_ref_var){

    // Exercise 1.1: 

    // mean point

    // variance

    // scale

    return;
}


void ProcrustesAnalysis::AlignShape(Mat in_tgt, Mat& in_ref, Mat& out_aligned){

    // Exercise 1.2: 

    // get the means

    // get the variance

    // get the rotation matrix

    // perform inverse transformation

    // copy to output

    return;
}


float ProcrustesAnalysis::ComputeAvgError(Mat& in_newdata){

    // Exercise 1.3: 

    float error = 0.0;
    // get avg. error
    return error;
}


void ProcrustesAnalysis::displayShape(Mat& in_shapes, string header, int waitFlag){
    // init interm. parameters
    int scl=500;
    double maxval;
    RNG rng;
    Mat shapes = in_shapes.clone();
    minMaxLoc(shapes,NULL,&maxval,NULL,NULL);
    shapes *= (scl*0.8/maxval);

    Mat dispImg(scl,scl,CV_8UC3);
    Scalar color(0,0,0);
    dispImg.setTo(color);
    int lstx=shapes.rows/2-1;

    if(0==waitFlag){
        color[0]=20; color[1]=40; color[2]=180;
    }

    // draw each input shape in a different color
    for(int cidx=0; cidx<shapes.cols; ++cidx){
        if(1==waitFlag){
            color[0]=rng.uniform(0,256); color[1]=rng.uniform(0,256); color[2]=rng.uniform(0,256);
        }
        for(int ridx=0; ridx<lstx-1; ++ridx){
            Point2i startPt(shapes.at<float>(ridx,cidx),shapes.at<float>(ridx+lstx+1,cidx));
            Point2i endPt(shapes.at<float>(ridx+1,cidx),shapes.at<float>(ridx+lstx+2,cidx));
            line(dispImg,startPt,endPt,color,2);
        }
        imshow(header.c_str(),dispImg);
        waitKey(150);
    }
    if(1==waitFlag){
        cout<<"press any key to continue..."<<endl;
        waitKey(10);
    }
}


/////////////////////////////////////////////////
////  class declaration for tasks 2 and 3  //////
/////////////////////////////////////////////////
class ShapeModel{
public:
    ShapeModel(){rng(10); scl=400;};
    ~ShapeModel(){};
    void loadData(const string& fileLoc, Mat& data);
    void trainModel();
    void displayModel();
    void inference();
    /* utilities */
    void displayShape(Mat& shapes, string header, int waitFlag=0);
    /* variables */
    Mat trainD;
    Mat testD;

private:
    Mat meanShape;
    Mat prinComp;
    Mat prinVal;
    RNG rng;
    int scl;
};

void ShapeModel::loadData(const string& fileLoc, Mat& data){

    // check if file exists
    ifstream iStream(fileLoc.c_str());
    if(!iStream){
        cout<<"file for load data cannot be found"<<endl;
        exit(-1);
    }

    // read aligned hand shapes
    int rows, cols;
    iStream >> rows;
    iStream >> cols;
    data.create(rows,cols,CV_32F);
    data.setTo(0);
    float *dptr;
    for(int ridx=0; ridx<data.rows; ++ridx){
        dptr = data.ptr<float>(ridx);
        for(int cidx=0; cidx<data.cols; ++cidx, ++dptr){
            iStream >> *dptr;
        }
    }
    iStream.close();
}

void ShapeModel::trainModel(){

    // Exercise 2.1: 

    // find mean

    // find covariance

    // find eigenvectors and eigen values

    // store principle components
    return;
}

void ShapeModel::displayModel(){

    // Exercise 2.2: 

    // visualize weights 

    return;
}


void ShapeModel::inference(){

    // Exercise 3: 

    // decomposition

    // and reconstruction

    return;

}

void ShapeModel::displayShape(Mat& shapes,string header, int waitFlag){
    // init interm. parameters
    Mat dispImg(scl,scl,CV_8UC3);
    Scalar color(0,0,0);
    dispImg.setTo(color);
    int lstx=shapes.rows/2-1;

    if(0==waitFlag){
        color[0]=20; color[1]=40; color[2]=180;
    }

    // draw each input shape in a different color
    for(int cidx=0; cidx<shapes.cols; ++cidx){
        if(1==waitFlag){
            color[0]=rng.uniform(0,256); color[1]=rng.uniform(0,256); color[2]=rng.uniform(0,256);
        }
        for(int ridx=0; ridx<lstx-1; ++ridx){
            Point2i startPt(shapes.at<float>(ridx,cidx),shapes.at<float>(ridx+lstx+1,cidx));
            Point2i endPt(shapes.at<float>(ridx+1,cidx),shapes.at<float>(ridx+lstx+2,cidx));
            line(dispImg,startPt,endPt,color,2);
        }
        imshow(header.c_str(),dispImg);
        waitKey(150);
    }
    if(1==waitFlag){
        cout<<"press any key to continue..."<<endl;
        waitKey(0);
    }
}



int main(){

    // Procrustes Analysis
    cout <<  "Procrustes Analysis:" << endl;
    ProcrustesAnalysis proc;
    proc.LoadData(file_for_procustes);
    proc.AlignData();

    // Shape Analysis
    // training procedure
    cout <<  "Shape Analysis - Training:" << endl;
    ShapeModel model;
    model.loadData(train_file_shape,model.trainD);
    model.displayShape(model.trainD,string("trainingShapes"));
    model.trainModel();
    model.displayModel();

    // testing procedure
    cout <<  "Shape Analysis - Testing:" << endl;
    model.loadData(test_file_shape,model.testD);
    model.inference();

    cout <<                                                                                                   endl;
    cout << "///////////////////////////////" << endl;
    cout << "///////////////////////////////" << endl;
    cout << "//////////    END    //////////" << endl;
    cout << "///////////////////////////////" << endl;
    cout << "///////////////////////////////" << endl;
    cout <<                                                                                                   endl;

    cout << "exiting code" << endl;
    return 0;
}
