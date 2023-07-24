#include "LSGD.h"
LSGD::LSGD(Mat img): img_(img){
        cvtColor(img_, gray_img_, COLOR_BGR2GRAY);
        img_.copyTo(seg_DR_);
        Irow_ = img_.rows;
        Icol_ = img_.cols;
    patch_size_ = 50;
        seg_row_ = Irow_ / (patch_size_ * 2) + 1;
        seg_col_ = Icol_ / (patch_size_ * 2) + 1;
        pixInd_Dist_.resize(Irow_, Icol_);
        pixInd_Dist_.fill(make_pair(-1, -1));
    seg_gray_.resize(seg_row_, seg_col_);
        kernel.resize(5, 5);
        kernel << 0, 0, 1, 0, 0
        , 0, 1, 2, 1, 0
        , 1, 2, -16, 2, 1
        , 0, 1, 2, 1, 0
        , 0, 0, 1, 0, 0;
        srand(time(nullptr));
}

//Center point position adjustment
void LSGD::adjustKPoint(Point &kp){
    if(kp.x <= 0 || kp.y <= 0 || kp.x >= Icol_-1 || kp.y >= Irow_-1){
        cerr << "KeyPoint outside the range." << endl;
        return;
    }
    Point new_kp = kp;
    float G = FLT_MAX;
    for(int i=-1; i<=1; ++i){
        for(int j=-1; j<=1; ++j){
            if(i == 0 && j == 0) continue;
            float LG = norm((img_.at<Vec3b>(kp) - img_.at<Vec3b>(kp + Point{i, j}))) / sqrt(i*i + j*j);
            if(G > LG){
                G = LG;
                new_kp = kp + Point{i, j};
            }
        }
    }
    kp = new_kp;
}

Eigen::Matrix<Point, Eigen::Dynamic, Eigen::Dynamic> LSGD::classification(const Eigen::Matrix<Point, Eigen::Dynamic, Eigen::Dynamic> &kPoints){
    float Srow = Irow_ / seg_row_, Scol = Icol_ / seg_col_;
    for(int r = 0; r < seg_row_; ++r){
        for(int c = 0; c < seg_col_; ++c){
            uint16_t localrow = kPoints(r, c).y, localcol = kPoints(r, c).x;
            for(int sr = -Srow; sr <= Srow; ++ sr){
                for(int sc = -Scol; sc <= Scol; ++ sc){
                    int newr = localrow + sr, newc = localcol + sc;
                    if(newr < 0 || newr >= Irow_ || newc < 0 || newc >= Icol_) continue;
//                    int distIn = norm(img_.at<Vec3b>(newr, newc) - img_.at<Vec3b>(localrow, localcol));
                    int distIn = fabs(gray_img_.at<uchar>(newr, newc) - gray_img_.at<uchar>(localrow, localcol));
                    float distEu = sqrt(sc*sc + sr*sr);
                    float dist = sqrt(distIn*distIn/intensity_scale_ + distEu*distEu/distance_scale_);
                    if(pixInd_Dist_(newr, newc).first == -1 || dist < pixInd_Dist_(newr, newc).second) {
                        pixInd_Dist_(newr, newc) = make_pair((r * seg_col_ + c), dist);
                    }
                }
            }
        }
    }
    //K-means classification of nearby pixels based on fuse distance
    vector<vector<Point>> kmeansMat(seg_row_*seg_col_);
    for(int r = 0; r < Irow_; ++ r){
        for(int c = 0; c < Icol_; ++c){
            kmeansMat[pixInd_Dist_(r, c).first].push_back(Point {c, r});
        }
    }
    //Obtain the center point of each grid after iteration
    Eigen::Matrix<Point, Eigen::Dynamic, Eigen::Dynamic> newKPoints;
    newKPoints.resize(seg_row_, seg_col_);
    uint64_t pixnum = 0;
    seg_gray_.fill(vector<uchar>());
    for(int i = 0; i < seg_row_*seg_col_; ++i){
        Point point{0, 0};
        for(auto p : kmeansMat[i]){
            point += p;
            seg_gray_(i/seg_col_, i%seg_col_).push_back(gray_img_.at<uchar>(p));
        }
        point.x /= kmeansMat[i].size();
        point.y /= kmeansMat[i].size();
        newKPoints(i/seg_col_, i%seg_col_) = point;
    }
    return newKPoints;
}

//image segmentation
void LSGD::segImg() {
    Eigen::Vector2f sRIO{(float )Icol_/seg_col_, (float )Irow_/seg_row_};
    kPoints.resize(seg_row_, seg_col_);
    for(int i = 0; i < seg_col_; ++i){
        for(int j = 0; j < seg_row_; ++j){
            Point p{ int((i+0.5)*sRIO.x()), int((j+0.5)*sRIO.y())};
            adjustKPoint(p);
            kPoints(j, i) = p;
        }
    }
    while(super_pixel_iter_--){
        kPoints = classification(kPoints);
    }
}

void LSGD::grayHist() {
    hist_gray_.resize(seg_row_*seg_col_);
    #pragma omp parallel for
    for(int i = 0; i < seg_row_*seg_col_; ++i)
        hist_gray_[i] = calHistVec(seg_gray_(i/seg_col_, i%seg_col_));
//    int B=rand()%100, G=rand()%150, R=rand()%220;
    //Visualization of segmentation results
    int B=80, G=120, R=200;
    for(int r = 0; r < Irow_; ++r){
        for(int c = 0; c < Icol_; ++c){
            uint16_t classedVal = pixInd_Dist_(r, c).first + 1;
            seg_DR_.at<Vec3b>(r, c) =
                    Vec3b(int(classedVal*B)%255,
                          int(classedVal*G)%255,
                          int(classedVal*R)%255);
        }
    }
    addWeighted(img_, 0.5, seg_DR_, 0.7, 0, seg_DR_);
}

void LSGD::conNet(){
    Eigen::Matrix<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Dynamic, Eigen::Dynamic> patches;
    patches.resize(seg_row_, seg_col_);
    static Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> descConNet;
    desc_ConNet_.resize(seg_row_ * seg_col_);
    for(int r = 0; r < seg_row_; ++r){
        for(int c = 0; c < seg_col_; ++c){
            Point cerP = kPoints(r, c);
            if(cerP.x < patch_size_)
                cerP.x = patch_size_;
            if(cerP.y < patch_size_)
                cerP.y = patch_size_;
            if(cerP.x + patch_size_ >= Icol_)
                cerP.x = Icol_ - patch_size_ - 1;
            if(cerP.y + patch_size_ >= Irow_)
                cerP.y = Irow_ - patch_size_ - 1;
//            cout << "r c " << cerP << endl;
            cnn_patches.push_back(Rect(cerP.x-patch_size_, cerP.y-patch_size_, patch_size_+patch_size_, patch_size_+patch_size_));
            Mat Patch(gray_img_, cnn_patches.back());

            cv2eigen(Patch, patches(r,c));
//          decsConNet = poolLayer(poolLayer(convLayer(patches(r, c), kernel)));
            descConNet = poolLayer(convLayer(patches(r, c), kernel));
//          descConNet = convLayer(patches(r, c), kernel);
            uint16_t conR = descConNet.rows(), conC = descConNet.cols();
            for(int i = 0; i < conR; ++i){
                for(int j = 0; j < conC; ++j){
                    desc_ConNet_[r*seg_col_ + c].push_back(descConNet(i, j));
                }
            }
        }
    }
    Mat cnnPatch;
    img_.copyTo(cnnPatch);
    int B=80, G=120, R=200;
    for(auto patch : cnn_patches){
        uint16_t classedVal = pixInd_Dist_(patch.y, patch.x).first + 1;
        cv::rectangle(cnnPatch, patch, Scalar(int(classedVal*B)%255,
                                              int(classedVal*G)%255,
                                              int(classedVal*R)%255), 3, LINE_8,0);
    }
    imwrite((seg_img_path_ + "cnnPatches/" + to_string(img_idx_) + "cp" + ".png"), cnnPatch);
//    imshow("cnnPatch", cnnPatch);
}

void LSGD::resetPatchSize(uint16_t size){
    patch_size_ = size;
    seg_row_ = Irow_ / (patch_size_ * 2) + 1;
    seg_col_ = Icol_ / (patch_size_ * 2) + 1;
    pixInd_Dist_.resize(Irow_, Icol_);
    pixInd_Dist_.fill(make_pair(-1, -1));
    seg_gray_.resize(seg_row_, seg_col_);
}
