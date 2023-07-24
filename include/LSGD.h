#ifndef _LSGD_H
#define _LSGD_H
#include <queue>
#include "tools.h"

class LSGD {
public:
    Mat img_;
    Mat gray_img_, seg_DR_;
    vector<vector<int>> desc_ConNet_;
    vector<vector<uint32_t>> hist_gray_;
    int super_pixel_iter_;
    float intensity_scale_, distance_scale_;
    uint16_t patch_size_;
    string seg_img_path_;
    int img_idx_;

    LSGD(Mat img);
    ~LSGD(){};

    void adjustKPoint(Point &kp);
    Eigen::Matrix<Point, Eigen::Dynamic, Eigen::Dynamic> classification(const Eigen::Matrix<Point, Eigen::Dynamic, Eigen::Dynamic> &kPoints);
    void segImg();
    void grayHist();
    void conNet();
    void resetPatchSize(uint16_t size);

private:
    uint32_t Irow_, Icol_;
    uint16_t seg_row_, seg_col_;
    Eigen::Matrix<pair<uint64_t, float>, Eigen::Dynamic, Eigen::Dynamic> pixInd_Dist_;
    Eigen::Matrix<Point, Eigen::Dynamic, Eigen::Dynamic> kPoints;
    vector<Rect> cnn_patches;

    Eigen::Matrix<Mat, Eigen::Dynamic, Eigen::Dynamic> seg_imgs_;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> gray_imgs_;
    Eigen::Matrix<vector<uchar>, Eigen::Dynamic, Eigen::Dynamic> seg_gray_;
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> kernel;
};

#endif //_LSGD_H
