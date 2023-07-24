#include "DynamicNode.h"

DynamicNode::DynamicNode(std::string file_path) {
    file_path_ = file_path;
    getFileNames(file_path, imgs_path_);
    sort(imgs_path_.begin(), imgs_path_.end(), less_equal<string>());
    imgs_num_ = imgs_path_.size();
    imgs_hist_grays_.resize(imgs_num_);
    imgs_hist_connet_.resize(imgs_num_);
    sim_score_mat_.resize(imgs_num_, imgs_num_);
    init_state_ = true;
}

void DynamicNode::getFileNames(string path, vector<string>& filenames) {
    DIR *pDir;
    struct dirent *ptr;
    if (!(pDir = opendir(path.c_str()))){
        cerr << "No " << path << " find" << endl;
        return;
    }
    while ((ptr = readdir(pDir)) != 0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            filenames.push_back(ptr->d_name);
    }
    closedir(pDir);
}

// LSGD
vector<pair<uint32_t, float>> DynamicNode::getGraySimIdx(const vector<vector<uint32_t>> &hist_gray){
    vector<vector<uint32_t>> last_hist_gray = imgs_hist_grays_[!init_state_ ? curr_idx_-1 : 0];
    //Calculation of candidate loop closure frame threshold
    double thred_score = similar(hist_gray, last_hist_gray) * 0.9;
    vector<pair<uint32_t, float>> cand_idxs;
//    similarScoreMat_.resize(currInx_ + 1, currInx_ + 1);
    for(uint32_t i = 0; i <= curr_idx_; ++i){
        double i_score = similar(hist_gray, imgs_hist_grays_[i]);
        sim_score_mat_(i, curr_idx_) = i_score;
        sim_score_mat_(curr_idx_, i) = i_score;
        if(curr_idx_-i > disc_num_ && i_score > thred_score){
            cand_idxs.push_back(make_pair(i, i_score));
        }
    }
    //Sort the obtained candidate frames in descending order based on similarity
    sort(cand_idxs.begin(), cand_idxs.end(), [](const pair<int, float> &a, const pair<int, float> &b){
        return a.second > b.second;
    });
    return cand_idxs;
}

vector<pair<uint32_t, float>> DynamicNode::getConNetSimIdx(const vector<vector<int>> &HistCovNet){
    vector<vector<int>> lastHistConNet = imgs_hist_connet_[!init_state_? curr_idx_-1 : 0];
    double thred_score = similar(HistCovNet, lastHistConNet) * 0.9;
    vector<pair<uint32_t, float>> cand_idxs;
//    similarScoreMat_.resize(currInx_ + 1, currInx_ + 1);
    for(uint32_t i = 0; i <= curr_idx_; ++i){
        double i_score = similar(HistCovNet, imgs_hist_connet_[i]);
        sim_score_mat_(i, curr_idx_) = i_score;
        sim_score_mat_(curr_idx_, i) = i_score;
        if(curr_idx_-i > disc_num_ && i_score > thred_score){
            cand_idxs.push_back(make_pair(i, i_score));
        }
    }
    sort(cand_idxs.begin(), cand_idxs.end(), [](const pair<int, float> &a, const pair<int, float> &b){
        return a.second > b.second;
    });
    return cand_idxs;
}

// LSGD + dynamic node
vector<pair<uint32_t, float>> DynamicNode::getGroupGraySimIdx(const vector<vector<uint32_t>> &HistGray){
    vector<vector<uint32_t>> lastHistGray = imgs_hist_grays_[!init_state_ ? curr_idx_-1 : 0];

    if(init_state_) group_similar_.push_back(vector<uint32_t>{0});
    int group_size = group_similar_.size();
    double thred_score = similar(HistGray, lastHistGray) * threshold_0_;
    vector<pair<uint32_t, float>> cand_idxs;
    //Determine which node the current frame should be located at with greater similarity
    for(uint32_t i = 0; i < group_size; ++i){
        float GiScore = similar(HistGray, imgs_hist_grays_[group_similar_[i][0]]);
        if(curr_idx_ - group_similar_[i][0] > disc_num_ && GiScore > thred_score * threshold_1_){
            float ave_score = 0.0;
            vector<float> GiScores;
            int GiSize = group_similar_[i].size();
            //#pragma omp parallel for
            for(uint32_t gi = 0; gi < GiSize; ++gi){
                float giscore = similar(HistGray, imgs_hist_grays_[group_similar_[i][gi]]);
                GiScores.push_back(giscore);
            }
            ave_score = accumulate(GiScores.begin(), GiScores.end(), 0.0) / GiSize;
            if(ave_score > thred_score * threshold_2_){
//                #pragma omp parallel for
                for(int ci = 0; ci < GiSize; ++ci){
                    cand_idxs.push_back(make_pair(group_similar_[i][ci], GiScores[ci]));
                    sim_score_mat_(group_similar_[i][ci], curr_idx_) = GiScores[ci];
                    sim_score_mat_(curr_idx_, group_similar_[i][ci]) = GiScores[ci];
                    sim_score_mat_(curr_idx_, curr_idx_) = 1.0;
                }
                group_similar_[i].push_back(curr_idx_);
                break;
            }
        }
    }
    //Build new dynamic node
    if(cand_idxs.empty()){
        group_similar_.push_back(vector<uint32_t>{(uint32_t)curr_idx_});
        sim_score_mat_(curr_idx_, curr_idx_) = 1.0;
        return cand_idxs;
    }
    sort(cand_idxs.begin(), cand_idxs.end(), [](const pair<int, float> &a, const pair<int, float> &b){
        return a.second > b.second;
    });
    return cand_idxs;
}

vector<pair<uint32_t, float>> DynamicNode::getGroupConvNetSimIdx(const vector<vector<int>> &HistConNet){
    vector<vector<int>> lastHistConNet = imgs_hist_connet_[!init_state_? curr_idx_-1 : 0];
    if(init_state_) group_similar_.push_back(vector<uint32_t>{0});
    int group_size = group_similar_.size();
    double thred_score = similar(HistConNet, lastHistConNet) * threshold_0_;
    vector<pair<uint32_t, float>> cand_idxs;
    for(uint32_t i = 0; i < group_size; ++i){
        float GiScore = similar(HistConNet, imgs_hist_connet_[group_similar_[i][0]]);
        if(curr_idx_ - group_similar_[i][0] > disc_num_ && GiScore > thred_score * threshold_1_){
            float ave_score = 0.0;
            vector<float> GiScores;
            int GiSize = group_similar_[i].size();
            //#pragma omp parallel for
            for(uint32_t gi = 0; gi < GiSize; ++gi){
                float giscore = similar(HistConNet, imgs_hist_connet_[group_similar_[i][gi]]);
                GiScores.push_back(giscore);
            }
            ave_score = accumulate(GiScores.begin(), GiScores.end(), 0.0) / GiSize;
            if(ave_score > thred_score * threshold_2_){
//                #pragma omp parallel for
                for(int ci = 0; ci < GiSize; ++ci){
                    cand_idxs.push_back(make_pair(group_similar_[i][ci], GiScores[ci]));
                    sim_score_mat_(group_similar_[i][ci], curr_idx_) = GiScores[ci];
                    sim_score_mat_(curr_idx_, group_similar_[i][ci]) = GiScores[ci];
                }
                group_similar_[i].push_back(curr_idx_);
                break;
            }
        }
    }
    if(cand_idxs.empty()){
        group_similar_.push_back(vector<uint32_t>{(uint32_t)curr_idx_});
        return cand_idxs;
    }
    sort(cand_idxs.begin(), cand_idxs.end(), [](const pair<int, float> &a, const pair<int, float> &b){
        return a.second > b.second;
    });
    return cand_idxs;
}

void DynamicNode::getConNetSimIdx(const vector<vector<int>> &descConNet, const vector<pair<uint32_t, float>> &candidImg){
    if(candidImg.empty()) return;

    vector<vector<int>> lastHistConNet = imgs_hist_connet_[curr_idx_-1];
    int candidNum = candidImg.size();
    vector<pair<uint32_t, float>> cand_idxs = candidImg;
    for(uint32_t i = 0; i < candidNum; ++i){
        double iScore = similar(descConNet, imgs_hist_connet_[candidImg[i].first]);
        sim_score_mat_(curr_idx_, candidImg[i].first) *= (1.0 + iScore);
        sim_score_mat_(candidImg[i].first, curr_idx_) *= (1.0 + iScore);
    }
    return ;
}

void DynamicNode::getGraySimIdx(const vector<vector<uint32_t>> &descGray, const vector<pair<uint32_t, float>> &candidImg){
    if(candidImg.empty()) return;

    vector<vector<uint32_t>> lastHistGray = imgs_hist_grays_[curr_idx_-1];
    int candidNum = candidImg.size();
    vector<pair<uint32_t, float>> cand_idxs = candidImg;
    for(uint32_t i = 0; i < candidNum; ++i){
        double iScore = similar(descGray, imgs_hist_grays_[candidImg[i].first]);
        sim_score_mat_(curr_idx_, candidImg[i].first) *= (1.0 + iScore);
        sim_score_mat_(candidImg[i].first, curr_idx_) *= (1.0 + iScore);
    }
    return ;
}