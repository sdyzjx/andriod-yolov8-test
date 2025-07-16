//
// Created by 张靖轩 on 2025/7/15.
//

#ifndef YOLOV8_YOLO_H
#define YOLOV8_YOLO_H
#include <opencv2/core/core.hpp>

#include <net.h>
struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};
struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};

class YOLO {
public:
    YOLO();
    int load(int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);
    int load(AAssetManager* mgr, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);
    int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f);
    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

private:
    ncnn::Net yolo;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif //YOLOV8_YOLO_H
