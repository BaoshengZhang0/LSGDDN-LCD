#ifndef _TOOLS_H
#define _TOOLS_H

#include <vector>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
using namespace Eigen;

template <typename T>
void cv2Eigen(const cv::Mat &mat, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix);

template <typename T>
void eigen2Cv(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix, cv::Mat &mat);

float means(const vector<float> v);

float variance(const vector<float> v);

Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> convLayer(const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &input,
               const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &kernel, const int step = 1);

Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> poolLayer(const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &input);

vector<uint32_t> calHistVec(vector<uchar> vec);

float similar(const vector<vector<int>> &decs1, const vector<vector<int>> &decs2);
double similar(const vector<vector<uint32_t>> &decs1, const vector<vector<uint32_t>> &decs2);

#endif //_TOOLS_H
