#ifndef _DYNAMIC_NODE_H
#define _DYNAMIC_NODE_H
#include <sstream>
#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <vector>
#include <string.h>
#include "tools.h"

class DynamicNode {
public:
    bool init_state_;
    int imgs_num_;
    int curr_idx_;
    int disc_num_;
    float threshold_0_;
    float threshold_1_;
    float threshold_2_;
    string file_path_;
    vector<string> imgs_path_;
    vector<vector<vector<uint32_t>>> imgs_hist_grays_;
    vector<vector<vector<int>>> imgs_hist_connet_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sim_score_mat_;

    DynamicNode(string filePath);
    ~DynamicNode(){};
    vector<pair<uint32_t, float>> getGraySimIdx(const vector<vector<uint32_t>> &HistGray);
    vector<pair<uint32_t, float>> getConNetSimIdx(const vector<vector<int>> &HistCovNet);
    vector<pair<uint32_t, float>> getGroupGraySimIdx(const vector<vector<uint32_t>> &HistGray);
    vector<pair<uint32_t, float>> getGroupConvNetSimIdx(const vector<vector<int>> &HistGray);
    void getConNetSimIdx(const vector<vector<int>> &descConNet, const vector<pair<uint32_t, float>> &candidImg);
    void getGraySimIdx(const vector<vector<uint32_t>> &descGray, const vector<pair<uint32_t, float>> &candidImg);

private:
    vector<vector<uint32_t>> group_similar_;
    void getFileNames(string path,vector<string>& filenames);
};

#endif //_DYNAMIC_NODE_H
