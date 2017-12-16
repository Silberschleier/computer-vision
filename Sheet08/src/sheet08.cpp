#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    ///=======================================================///
    ///                        Task-1                         ///
    ///=======================================================///
    int neighbouring_matches = 2;
    float threshold = 0.4;

    /// read and show images
    Mat image1 = imread("./images/mountain1.png", IMREAD_COLOR);
    Mat image2 = imread("./images/mountain2.png", IMREAD_COLOR);
    imshow("Image-1", image1);
    imshow("Image-2", image2);
    waitKey(0);
    destroyAllWindows();

    /// compute keypoints and descriptors
    SIFT sift;
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    sift(image1, Mat(), keypoints1, descriptors1, false);
    sift(image2, Mat(), keypoints2, descriptors2, false);

    /// show keypoints
    Mat image_keypoints1, image_keypoints2;
    drawKeypoints(image1, keypoints1, image_keypoints1);
    drawKeypoints(image2, keypoints2, image_keypoints2);
    imshow("Keypoints-1", image_keypoints1);
    imshow("Keypoints-2", image_keypoints2);

    /// compute nearest matches
    BFMatcher matcher;
    vector<vector<DMatch>> matches_1to2;
    matcher.knnMatch(descriptors1, descriptors2, matches_1to2, neighbouring_matches);


    /// filter matches by ratio test
    vector<DMatch> filtered_matches_1to2;
    for ( auto nearest_matches : matches_1to2 ) {
        float ratio = nearest_matches[0].distance / nearest_matches[1].distance;
        if (ratio <= threshold) {
            filtered_matches_1to2.push_back(nearest_matches[0]);
        }
    }

    /// determine two-way matches
    vector<vector<DMatch>> matches_2to1;
    matcher.knnMatch(descriptors1, descriptors2, matches_2to1, neighbouring_matches);

    vector<DMatch> two_way_matches;
    for ( auto nearest_matches : matches_2to1 ) {
        auto best_match = nearest_matches[0];

        // Skip matches above threshold to reduce search space
        float ratio = best_match.distance / nearest_matches[1].distance;
        if (ratio > threshold) {
            continue;
        }

        // Check if match is also in 1to2 matches
        for ( auto match : filtered_matches_1to2 ) {
            if ( match.queryIdx == best_match.trainIdx ) {
                two_way_matches.push_back(match);
                break;
            }
        }
    }

    cout << "Size keypoints1: " << keypoints1.size() << endl;
    cout << "Size keypoints2: " << keypoints2.size() << endl;
    cout << "Size matches_1to2: " << matches_1to2.size() << endl;
    cout << "Size filtered_matches_1to2: " << filtered_matches_1to2.size() << endl;
    cout << "Size two_way_matches: " << two_way_matches.size() << endl;

    /// visualize matching key-points
    Mat img_matches;
    drawMatches(image1, keypoints1, image2, keypoints2, two_way_matches, img_matches);
    imshow("Matches", img_matches);


    waitKey(0);
    destroyAllWindows();


    ///=======================================================///
    ///                        Task-2                         ///
    ///=======================================================///

    /// Implement RANSAC here
    RNG rng(0xFFFFFFFF);
    float epsilon = 1;

    for (int i = 0; i < 100; i++) {
        // Select random pairs
        Point2f coords1[4], coords2[4];
        set<int> selected_matches;
        while (selected_matches.size() <= 4) {
            int rnd_index = rng.uniform(0, (int) two_way_matches.size());

            // Skip already chosen matches
            if (selected_matches.count(rnd_index) == 1) continue;
            selected_matches.insert(rnd_index);

            auto match = two_way_matches[rnd_index];
            coords1[selected_matches.size()-1] = keypoints1[match.queryIdx].pt;
            coords2[selected_matches.size()-1] = keypoints2[match.trainIdx].pt;
        }

        // Compute homography
        Mat homography = getPerspectiveTransform(coords1, coords2);

        // Compute inliers
        for ( auto match : two_way_matches ) {
            auto pt1 = keypoints1[match.queryIdx].pt;
            auto pt2 = keypoints2[match.trainIdx].pt;

            Mat mapped_pt = homography * Mat(Vec3f(pt1.x, pt1.y, 1)).t();

        }

    }

    ///  Transform and stitch the images here





    /// visualize stitched image
    waitKey(0);
    destroyAllWindows();

}
