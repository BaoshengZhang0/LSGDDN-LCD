#include <thread>
#include <time.h>
#include "yaml-cpp/yaml.h"
#include "LSGD.h"
#include "DynamicNode.h"

int main(int argc, char** argv) {
    YAML::Node config = YAML::LoadFile("../config/config.yaml");
    string img_path = config["SYSTEM"]["img_path"].as<string>();
    string score_path = config["SYSTEM"]["score_path"].as<string>();
    bool seg_show_flag = config["LSGD"]["seg_show_flag"].as<bool>();
    string seg_img_path = config["LSGD"]["seg_img_path"].as<string>();

    DynamicNode dynamic_node(img_path);
    dynamic_node.disc_num_ = config["LSGD"]["disc_num"].as<int>();
    dynamic_node.threshold_0_ = config["DYNAMIC_NODE"]["threshold_0"].as<float>();
    dynamic_node.threshold_1_ = config["DYNAMIC_NODE"]["threshold_1"].as<float>();
    dynamic_node.threshold_2_ = config["DYNAMIC_NODE"]["threshold_2"].as<float>();

    const uint16_t init_grid_size = (uint16_t)config["LSGD"]["init_grid_size"].as<int>();
    const int seg_iter = config["LSGD"]["seg_iter"].as<int>();
    const float intensity_scale = config["LSGD"]["intensity_scale"].as<float>();
    const float distance_scale = config["LSGD"]["distance_scale"].as<float>();
    ofstream fout(score_path);
    clock_t start_time=clock();
    for(int idx_img = 0; idx_img < dynamic_node.imgs_num_; ++idx_img){
        if(idx_img == 0) dynamic_node.init_state_ = true;
        else dynamic_node.init_state_ = false;

        Mat img = imread(dynamic_node.file_path_ + "/" + dynamic_node.imgs_path_[idx_img], IMREAD_COLOR);
        if(img.empty()){
            break;
        }
        LSGD *lsgd = new LSGD(img);
        lsgd->super_pixel_iter_ = seg_iter;
        lsgd->intensity_scale_ = intensity_scale;
        lsgd->distance_scale_ = distance_scale;
        lsgd->seg_img_path_ = seg_img_path;
        lsgd->img_idx_ = idx_img;
        lsgd->resetPatchSize(init_grid_size);
        lsgd->segImg();
        lsgd->grayHist();
        dynamic_node.imgs_hist_grays_[idx_img] = lsgd->hist_gray_;
        dynamic_node.curr_idx_ = idx_img;

        cout << "\b\b\b\b\b\b\b\b\r" << fixed << setprecision(3) << "\033[36m" << "Processing : " << double (idx_img+1)*100/dynamic_node.imgs_num_ << "%" << "\033[1m" << flush;
        // LSGD
         vector<pair<uint32_t, float>> candidImg = dynamic_node.getGraySimIdx(lsgd->hist_gray_);
        // LSGD + dynamic node
//        vector<pair<uint32_t, float>> candidImg = dynamic_node.getGroupGraySimIdx(lsgd->hist_gray_);
        if(seg_show_flag){
            imshow("Segmented image", lsgd->seg_DR_);
            imwrite((seg_img_path + to_string(idx_img)+".png"), lsgd->seg_DR_);
            waitKey(1);
        }
        delete lsgd;
    }
    clock_t end_time=clock();
    cout << endl << fixed << "Average execution time: " << "\033[33m" << (double)(end_time - start_time) / 1000 / dynamic_node.imgs_num_ << "ms" << "\033[1m" << endl;
    fout << dynamic_node.imgs_num_ << " " << dynamic_node.imgs_num_ << endl;
    fout << dynamic_node.sim_score_mat_ << endl;
    fout.close();
    return 0;
}



