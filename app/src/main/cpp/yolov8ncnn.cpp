//
// Created by 张靖轩 on 2025/7/15.
//

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolo.h"
#include "net.h"
#include "cpu.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define TAG "YoloV8Ncnn_Native"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}
//采用结构体形式，方便在java与cpp之间实现传参
struct NativeProcessor {
    ANativeWindow* window = nullptr;
    YOLO* yolo_detector = nullptr; // 你的YOLO检测器实例
    // 其他需要的成员变量
    int surface_width = 0;
    int surface_height = 0;
};

// 全局或静态变量来持有 NativeProcessor 实例 (需要考虑多实例场景)
static NativeProcessor* g_processor = nullptr;
static ncnn::Mutex g_lock; // 用于保护对 g_processor 的访问
extern "C" {
    JNIEXPORT jlong JNICALL
    Java_com_example_yolov8_VideoProcessor_initNative(JNIEnv *env, jobject thiz, jobject surface, jobject assetManager, jboolean use_gpu) {
        ncnn::MutexLockGuard guard(g_lock); //ncnn加锁，防止多线程同时初始化
        //防止重复初始化
        if (g_processor) {
            LOGE("Native processor already initialized!");
            if (g_processor->window) {
                ANativeWindow_release(g_processor->window);
                g_processor->window = nullptr;
            }

        }
        //创建新processor对象
        g_processor = new NativeProcessor(); //
        if (!g_processor) {
            LOGE("Failed to allocate NativeProcessor");
            return 0;
        }
        //实例化window对象
        g_processor->window = ANativeWindow_fromSurface(env, surface);
        if (!g_processor->window) {
            LOGE("Failed to get ANativeWindow from Surface");
            delete g_processor;
            g_processor = nullptr;
            return 0;
        }
        //实例化yolo检测器
        g_processor->yolo_detector = new YOLO();
        AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
        if (mgr == nullptr) {
            LOGE("assert frome java failed");
            return JNI_FALSE;
        }
        const int target_sizes[] = {
            320,
            320,
        };
        const float mean_vals[][3] = {
            {103.53f, 116.28f, 123.675f},
            {103.53f, 116.28f, 123.675f},
        };
        const float norm_vals[][3] = {
            { 1 / 255.f, 1 / 255.f, 1 / 255.f },
            { 1 / 255.f, 1 / 255.f, 1 / 255.f },
        };
        //加载yolo模型
        int ret = g_processor->yolo_detector->load(mgr, target_sizes[0], mean_vals[0], norm_vals[0], (int)use_gpu);
        if (ret) {
            LOGE("Failed to load YOLO model");
            ANativeWindow_release(g_processor->window);
            delete g_processor->yolo_detector;
            delete g_processor;
            g_processor = nullptr;
            return 0;
        }
        LOGD("Native processor initialized successfully.");
        return reinterpret_cast<jlong>(g_processor);
    }
    JNIEXPORT void JNICALL
    Java_com_example_yolov8_VideoProcessor_processFrameNative(JNIEnv *env, jobject thiz, jlong native_ptr,
                                                            jbyteArray frame_data, jint width, jint height, jlong timestamp) {
        ncnn::MutexLockGuard guard(g_lock);
        NativeProcessor* processor = reinterpret_cast<NativeProcessor*>(native_ptr);
        if (!processor || !processor->window || !processor->yolo_detector) {
            LOGE("Native processor not initialized or window/detector is null.");
            return;
        }

        jbyte* pixels = env->GetByteArrayElements(frame_data, nullptr);
        if (!pixels) {
            LOGE("Failed to get byte array elements");
            return;
        }

        cv::Mat yuv420_frame(height * 3 / 2, width, CV_8UC1, pixels);
        cv::Mat rgb_frame;
        cv::cvtColor(yuv420_frame, rgb_frame, cv::COLOR_YUV2RGB_NV12);


        std::vector<Object> objects;
        processor->yolo_detector->detect(rgb_frame, objects);
        processor->yolo_detector->draw(rgb_frame, objects);

        draw_fps(rgb_frame); // 确保 draw_fps 接受 cv::Mat&
        cv::Mat rgba_frame;
        cv::cvtColor(rgb_frame, rgba_frame, cv::COLOR_RGB2RGBA);
        ANativeWindow_Buffer buffer;

        if (processor->surface_width != width || processor->surface_height != height) {
            if (ANativeWindow_setBuffersGeometry(processor->window, width, height, WINDOW_FORMAT_RGBA_8888) < 0) {
                LOGE("Cannot set ANativeWindow buffer geometry");
                env->ReleaseByteArrayElements(frame_data, pixels, JNI_ABORT);
                return;
            }
            processor->surface_width = width;
            processor->surface_height = height;
            LOGD("Set ANativeWindow buffer geometry to %d x %d", width, height);
        }

        if (ANativeWindow_lock(processor->window, &buffer, nullptr) < 0) {
            LOGE("Cannot lock ANativeWindow");
            env->ReleaseByteArrayElements(frame_data, pixels, JNI_ABORT);
            return;
        }

        auto* dst_pixels = static_cast<uint8_t*>(buffer.bits);
        int dst_stride = buffer.stride * 4; // stride in bytes for RGBA_8888

        cv::Mat dst_mat(buffer.height, buffer.width, CV_8UC4, dst_pixels, dst_stride);
        rgba_frame.copyTo(dst_mat);

        if (ANativeWindow_unlockAndPost(processor->window) < 0) {
            LOGE("Cannot unlock ANativeWindow and post");
        }

        env->ReleaseByteArrayElements(frame_data, pixels, JNI_ABORT);
    }
    JNIEXPORT void JNICALL
    Java_com_example_yolov8_VideoProcessor_releaseNative(JNIEnv *env, jobject thiz, jlong native_ptr) {
        ncnn::MutexLockGuard guard(g_lock);
        NativeProcessor* processor = reinterpret_cast<NativeProcessor*>(native_ptr);
        if (processor) {
            if (processor->window) {
                ANativeWindow_release(processor->window);
                processor->window = nullptr;
            }
            if (processor->yolo_detector) {
                delete processor->yolo_detector;
                processor->yolo_detector = nullptr;
            }
            delete processor;
            if (g_processor == processor) {
                g_processor = nullptr;
            }
            LOGD("Native processor released.");
        }
    }
}

