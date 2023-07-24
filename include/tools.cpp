#include "tools.h"

// Convert matrix type
template <typename T>
void cv2Eigen(const cv::Mat &mat, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix){
    uint16_t MatRow = mat.rows, MatCol = mat.cols;
    matrix.resize(MatRow, MatCol);
    for(int i = 0; i < MatRow; ++i){
        for(int j = 0; j < MatCol; ++j){
            matrix(i, j) = static_cast<T>(mat.at<uchar>(i, j));
        }
    }
}

template <typename T>
void eigen2Cv(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix, cv::Mat &mat){
    uint16_t MatRow = matrix.rows(), MatCol = matrix.cols();
    mat.resize(MatRow, MatCol);
    for(int i = 0; i < MatRow; ++i){
        for(int j = 0; j < MatCol; ++j){
            mat.at<uchar>(i, j) = matrix(i, j);
        }
    }
}

float means(const vector<float> v){
    if(v.empty()) return 0;
    float sum = accumulate(v.begin(), v.end(), 0.0);
//    float val = 0.0;
    return sum / v.size();
}

float variance(const vector<float> v){
    if(v.empty()) return 0;
    float val = 0.0;
    float mean = means(v);
    for(auto i : v){
        val += (i - mean) * (i - mean);
    }
    return val / v.size();
}

Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> convLayer(const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &input,
                                                             const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &kernel, const int step){
    uint16_t inRow = input.rows(), inCol = input.cols();
    uint16_t kerRow = kernel.rows(), kerCol = kernel.cols();
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> output;
    output.resize(inRow - kerRow + 1, inCol - kerCol + 1);
    for(int i = 0; i <= inRow - kerRow; i += step){
        for(int j = 0; j <= inCol - kerCol; j += step){
            int_least64_t conval = 0;
            Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> patch = input.block(i, j, kerRow, kerCol);
            for(int kr = 0; kr < kerRow; ++kr){
                for(int kc = 0; kc < kerCol; ++kc){
                    conval += (patch(kr, kc) * kernel(kr, kc));
                }
            }
            output(i, j) = conval >= 0 ? conval : 0;
        }
    }
//    cout << kernel << endl;
    return output;
}

Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> poolLayer(const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &input){
    uint16_t inRow = input.rows(), inCol = input.cols();
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> output;
    output.resize(inRow/2, inCol/2);
    for(int r = 0; r < inRow; r += 2){
        for(int c = 0; c < inCol; c += 2){
            output(r/2, c/2) = max(max(input(r, c), input(r, c+1)),
                                   max(input(r+1, c), input(r+1, c+1)));
        }
    }
//    cout << output.rows() << " " << output.cols() << endl;
    return output;
}

//Histogram Calculation
vector<uint32_t> calHistVec(vector<uchar> vec){
    vector<uint32_t> Hvec(256);
    for(auto v : vec){
        ++Hvec[v];
    }
    return Hvec;
}
//Descriptor similarity calculation
float similar(const vector<vector<int>> &decs1, const vector<vector<int>> &decs2){

    if(decs1.size() != decs2.size()){
        cerr << "decs1.size != decs2.size" << endl;
        return 0;
    }
    float score = 0.0;

    for(uint32_t i = 0; i < decs1.size(); ++i){
        if(decs1[i].size() != decs2[i].size()){
            cerr << "decs1[" << i << "] and decs2 size different length" << endl;
            return 0;
        }
        float tempscore = 0.0;
        for(uint32_t j = 0; j < decs1[i].size(); ++j){
            tempscore += pow(decs1[i][j] - decs2[i][j], 2);
        }
        tempscore = sqrt(tempscore);
        tempscore /= norm(decs1[i])+1;
        score += tempscore;
    }
    return 1.0/(exp(score/decs1.size()));
}

double similar(const vector<vector<uint32_t>> &decs1, const vector<vector<uint32_t>> &decs2){
    if(decs1.size() != decs2.size()){
        cerr << "decs1.size != decs2.size" << endl;
        return 0;
    }
    double score = 0.0;
    double maxscore = 0.0;
    for(uint32_t i = 0; i < decs1.size(); ++i){
        if(decs1[i].size() != decs2[i].size()){
            cerr << "decs1[" << i << "] and decs2 size different length" << endl;
            return 0;
        }
        double tempscore = 0.0;

        for(uint32_t j = 0; j < decs1[i].size(); ++j){
//            tempscore += pow(min((uint32_t)5000, decs1[i][j]) - min((uint32_t)5000, decs2[i][j]), 1);
            tempscore += abs((int64)decs1[i][j] - (int64)decs2[i][j]);
            maxscore = maxscore < tempscore ? tempscore : maxscore;
        }
        tempscore = sqrt(tempscore);
        tempscore /= 256;
//        cout << tempscore << " ";
//        tempscore /= (norm(decs1[i]));
        score += tempscore;
    }
//    score -= (sqrt(maxscore)/256);
//    cout << log(score/decs1.size() + 1) << endl;
    return 1.0/(1 + score/decs1.size());
}

