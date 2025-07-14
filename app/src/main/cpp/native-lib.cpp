#include <jni.h>
#include <string>
#include <vector>
#include <algorithm> // Required for std::max/min

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

// --- Data Structures ---

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// For YOLOv8 post-processing
struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};


// --- Global Variables ---

static ncnn::Net yolov8;
static std::vector<std::string> class_names;
// 新增：内存池分配器，用于加速
static ncnn::UnlockedPoolAllocator blob_pool_allocator;
static ncnn::PoolAllocator workspace_pool_allocator;

// --- Helper Functions (from reference code, with modifications) ---

// Load labels from assets (your original function, works fine)
static int load_labels(AAssetManager* mgr, const char* filename) {
    AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    if (!asset) {
        __android_log_print(ANDROID_LOG_ERROR, "NCNN", "Failed to open label file: %s", filename);
        return -1;
    }
    class_names.clear(); // Clear old labels before loading
    off_t size = AAsset_getLength(asset);
    std::string buffer(size, ' ');
    AAsset_read(asset, &buffer[0], size);
    AAsset_close(asset);

    std::string line;
    std::istringstream iss(buffer);
    while (std::getline(iss, line)) {
        class_names.push_back(line);
    }
    __android_log_print(ANDROID_LOG_INFO, "NCNN", "Loaded %d class names.", class_names.size());
    return 0;
}

static float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides) {
    grid_strides.clear();
    for (int stride : strides) {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++) {
            for (int g0 = 0; g0 < num_grid_w; g0++) {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}

static void generate_proposals(const std::vector<GridAndStride>& grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects) {
    const int num_points = grid_strides.size();
    const int num_class = class_names.size(); // Use dynamically loaded class count
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++) {
        const float* scores = pred.row(i) + 4 * reg_max_1;

        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++) {
            if (scores[k] > score) {
                label = k;
                score = scores[k];
            }
        }
        float box_prob = sigmoid(score);

        if (box_prob >= prob_threshold) {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
            {
                ncnn::Layer* softmax = ncnn::create_layer("Softmax");
                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);
                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;
                softmax->create_pipeline(opt);
                softmax->forward_inplace(bbox_pred, opt);
                softmax->destroy_pipeline(opt);
                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++) {
                float dis = 0.f;
                const float* dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++) {
                    dis += l * dis_after_sm[l];
                }
                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;
            objects.push_back(obj);
        }
    }
}

// NMS and sorting functions (your original functions, work fine)
static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;
    while (i <= j) {
        while (objects[i].prob > p) i++;
        while (objects[j].prob < p) j--;
        if (i <= j) {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }
    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}

static void qsort_descent_inplace(std::vector<Object>& objects) {
    if (objects.empty()) return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static inline float intersection_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    const int n = objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = objects[i].rect.area();
    }
    for (int i = 0; i < n; i++) {
        const Object& a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = objects[picked[j]];
            // NMS should only apply to boxes of the same class
            if (a.label != b.label) continue;
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold) {
                keep = 0;
                break; // Exit inner loop early
            }
        }
        if (keep) {
            picked.push_back(i);
        }
    }
}


// Drawing function (your original function, works fine)
static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects) {
    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];

        // Use a simple color scheme for visualization
        cv::Scalar color( (obj.label * 60) % 255, (obj.label * 100) % 255, (obj.label * 140) % 255);

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


// --- JNI Functions ---

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_example_yolov8_MainActivity_initYolo(JNIEnv *env, jobject thiz, jobject assetManager, jboolean use_gpu) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (mgr == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "NCNN", "AAssetManager_fromJava failed.");
        return JNI_FALSE;
    }

    // --- 开始优化 ---

    // 1. 配置NCNN选项(Option)，这是性能优化的核心
    ncnn::Option opt;

    // 1.1 启用Vulkan GPU加速
    // NCNN_VULKAN 是编译时宏，确保你的NCNN库是支持Vulkan的版本
#if NCNN_VULKAN
    opt.use_vulkan_compute = use_gpu;
    __android_log_print(ANDROID_LOG_INFO, "NCNN", "Vulkan GPU acceleration set to: %d", (int)use_gpu);
#else
    __android_log_print(ANDROID_LOG_WARN, "NCNN", "NCNN Vulkan not compiled, defaulting to CPU.");
    opt.use_vulkan_compute = false;
#endif

    // 1.2 优化CPU线程数，使用大核心
    // set_cpu_powersave(2) 推荐在应用启动时全局设置一次即可
    ncnn::set_cpu_powersave(2);
    opt.num_threads = ncnn::get_big_cpu_count();
    __android_log_print(ANDROID_LOG_INFO, "NCNN", "Using %d big CPU cores.", opt.num_threads);


    // 1.3 绑定内存池分配器
    // 在加载模型前清空并设置分配器
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    opt.blob_allocator = &blob_pool_allocator;
    opt.workspace_allocator = &workspace_pool_allocator;

    // --- 优化结束 ---


    // 2. 加载模型（将优化好的opt传入）
    yolov8.opt = opt;

    const char* model_param = "yolov8n.param";
    const char* model_bin = "yolov8n.bin";
    if (yolov8.load_param(mgr, model_param) != 0 || yolov8.load_model(mgr, model_bin) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "NCNN", "Failed to load ncnn model.");
        return JNI_FALSE;
    }
    __android_log_print(ANDROID_LOG_INFO, "NCNN", "NCNN model loaded successfully.");

    // 3. 加载标签文件
    const char* label_file = "label.txt";
    if (load_labels(mgr, label_file) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "NCNN", "Failed to load labels.");
        return JNI_FALSE;
    }

    return JNI_TRUE;
}

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

    // 1. Convert Bitmap to cv::Mat
    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);

    // --- START: REVISED PRE-PROCESSING & INFERENCE ---

    // 2. Pre-processing
    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

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

    int wpad = (img_w + 31) / 32 * 32 - img_w;
    int hpad = (img_h + 31) / 32 * 32 - img_h;
    ncnn::Mat in_pad;
    // Use 114.f for padding as in your original code, which is common for YOLOv8
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    // 3. Inference
    ncnn::Extractor ex = yolov8.create_extractor();
    ex.input("images", in_pad);
    ncnn::Mat out;
    ex.extract("output", out);

    // 4. Post-processing (using the correct YOLOv8 logic)
    std::vector<Object> proposals;
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    generate_proposals(grid_strides, out, prob_threshold, proposals);

    // NMS
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    // Adjust boxes back to original image coordinates
    std::vector<Object> objects;
    for (int i = 0; i < picked.size(); i++) {
        Object obj = proposals[picked[i]];

        // Adjust for padding and scaling
        float x0 = (obj.rect.x - (wpad / 2)) / scale;
        float y0 = (obj.rect.y - (hpad / 2)) / scale;
        float x1 = (obj.rect.x + obj.rect.width - (wpad / 2)) / scale;
        float y1 = (obj.rect.y + obj.rect.height - (hpad / 2)) / scale;

        // Clip to image boundaries
        x0 = std::max(std::min(x0, (float)(bgr.cols - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(bgr.rows - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(bgr.cols - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(bgr.rows - 1)), 0.f);

        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;

        objects.push_back(obj);
    }

    // --- END: REVISED LOGIC ---

    // 5. Draw results onto the BGR Mat
    draw_objects(bgr, objects);

    // 6. Convert the processed BGR Mat back to RGBA for the Bitmap
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);

    AndroidBitmap_unlockPixels(env, bitmap);

    return JNI_TRUE;
}

} // extern "C"