#include <jni.h>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// ncnn
#include "net.h"
#include "cpu.h"

// Android NDK Bitmap
#include <android/bitmap.h>
#include <android/log.h>
#include <android/asset_manager_jni.h>
// 结构体，用于存放检测结果
struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// 全局 YoloV8 推理器实例
static ncnn::Net yolov8;
// 标签
static std::vector<std::string> class_names;

// 从 assets 加载标签文件
static int load_labels(AAssetManager* mgr, const char* filename) {
    AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    if (!asset) {
        __android_log_print(ANDROID_LOG_ERROR, "NCNN", "Failed to open label file: %s", filename);
        return -1;
    }

    off_t size = AAsset_getLength(asset);
    std::string buffer(size, ' ');
    AAsset_read(asset, &buffer[0], size);
    AAsset_close(asset);

    // 按行解析
    std::string line;
    std::istringstream iss(buffer);
    while (std::getline(iss, line)) {
        class_names.push_back(line);
    }
    __android_log_print(ANDROID_LOG_INFO, "NCNN", "Loaded %d class names.", class_names.size());
    return 0;
}


// YOLOv8 后处理函数
static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;
        while (faceobjects[j].prob < p)
            j--;
        if (i <= j) {
            std::swap(faceobjects[i], faceobjects[j]);
            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects) {
    if (faceobjects.empty())
        return;
    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static inline float intersection_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    const int n = faceobjects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[picked[j]];
            if (a.label != b.label)
                continue;
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

// 绘制检测框
static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects) {
    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];

        cv::Scalar color = cv::Scalar(255, 0, 0); // BGR
        float C = obj.prob;
        int color_index = obj.label % 80;
        color = cv::Scalar(255, 0, 0);

        cv::rectangle(bgr, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > bgr.cols) x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), color, -1);
        cv::putText(bgr, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}


// --- JNI 函数 ---

extern "C" {

// 初始化函数，加载模型
JNIEXPORT jboolean JNICALL
Java_com_example_yolov8_MainActivity_initYolo(JNIEnv *env, jobject thiz, jobject assetManager) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (mgr == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "NCNN", "AAssetManager_fromJava failed.");
        return JNI_FALSE;
    }

    // 1. 加载模型
    const char* model_param = "yolov8s.ncnn.param";
    const char* model_bin = "yolov8s.ncnn.bin";
    if (yolov8.load_param(mgr, model_param) != 0 || yolov8.load_model(mgr, model_bin) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "NCNN", "Failed to load ncnn model.");
        return JNI_FALSE;
    }
    __android_log_print(ANDROID_LOG_INFO, "NCNN", "NCNN model loaded successfully.");


    // 2. 加载标签
    const char* label_file = "label.txt";
    if (load_labels(mgr, label_file) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "NCNN", "Failed to load labels.");
        return JNI_FALSE;
    }

    return JNI_TRUE;
}


// 检测函数，处理每一帧
JNIEXPORT jboolean JNICALL
Java_com_example_yolov8_MainActivity_detect(JNIEnv *env, jobject thiz, jobject bitmap) {
    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "NCNN", "AndroidBitmap_getInfo failed");
        return JNI_FALSE;
    }
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        __android_log_print(ANDROID_LOG_ERROR, "NCNN", "Bitmap format is not RGBA_8888");
        return JNI_FALSE;
    }
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "NCNN", "AndroidBitmap_lockPixels failed");
        return JNI_FALSE;
    }

    // 1. 将 Bitmap 转换为 cv::Mat
    cv::Mat img(info.height, info.width, CV_8UC4, pixels);
    cv::Mat bgr;
    cv::cvtColor(img, bgr, cv::COLOR_RGBA2BGR);

    // 2. 预处理
    const int target_size = 640;
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    float scale = 1.f;
    if (img_w > img_h) {
        scale = (float)target_size / img_w;
        img_w = target_size;
        img_h = img_h * scale;
    } else {
        scale = (float)target_size / img_h;
        img_h = target_size;
        img_w = img_w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, img_w, img_h);

    int wpad = (target_size - img_w) / 2;
    int hpad = (target_size - img_h) / 2;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad, target_size - img_h - hpad, wpad, target_size - img_w - wpad, ncnn::BORDER_CONSTANT, 114.f);

    // 归一化
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    // 3. 推理
    ncnn::Extractor ex = yolov8.create_extractor();
    ex.input("images", in_pad);
    ncnn::Mat out;
    ex.extract("output0", out);

    // 4. 后处理
    std::vector<Object> proposals;
    const int num_grid = out.h;
    const int num_class = out.w - 4;

    for (int i = 0; i < num_grid; i++) {
        const float* p_i = out.row(i);
        const float* class_scores = p_i + 4;
        int label = 0;
        float max_score = 0.f;
        for (int k = 0; k < num_class; k++) {
            if (class_scores[k] > max_score) {
                max_score = class_scores[k];
                label = k;
            }
        }

        if (max_score > 0.25f) {
            float x_center = p_i[0];
            float y_center = p_i[1];
            float w = p_i[2];
            float h = p_i[3];

            float x0 = x_center - w * 0.5f;
            float y0 = y_center - h * 0.5f;

            Object obj;
            obj.rect.x = (x0 - wpad) / scale;
            obj.rect.y = (y0 - hpad) / scale;
            obj.rect.width = w / scale;
            obj.rect.height = h / scale;
            obj.label = label;
            obj.prob = max_score;
            proposals.push_back(obj);
        }
    }

    // NMS
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, 0.45f);

    int count = picked.size();
    std::vector<Object> objects(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];
    }

    // 5. 绘制结果到原始图像
    draw_objects(bgr, objects);

    // 6. 将处理后的 cv::Mat 转换回 Bitmap
    cv::cvtColor(bgr, img, cv::COLOR_BGR2RGBA);

    AndroidBitmap_unlockPixels(env, bitmap);

    return JNI_TRUE;
}

} // extern "C"