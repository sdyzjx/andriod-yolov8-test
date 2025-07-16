本项目是一个 Android 应用程序，演示了如何在 Android 设备上使用 NCNN 深度学习框架和预训练的 YOLOv8 模型对视频文件进行实时目标检测。应用允许用户选择本地视频文件，并在 SurfaceView 上播放视频的同时，通过 JNI 调用本地 C++ 代码进行目标检测，并将结果渲染到视频帧上。

## ✨ 项目功能

*   **视频选择:** 用户可以通过系统的文件选择器选择本地存储的视频文件进行处理。
*   **视频解码:** 使用 Android `MediaCodec` 和 `MediaExtractor` API 对选定的视频文件进行解码。
*   **实时目标检测:**
    *   将解码后的视频帧数据传递给通过 JNI 调用的本地 C++ NCNN 推理引擎。
    *   使用 YOLOv8 模型进行目标检测。
*   **结果渲染:** 检测结果（例如边界框和标签）通过本地 C++ 代码直接绘制到 Android 的 `Surface` 上。
*   **SurfaceView 显示:** 使用 `SurfaceView` 高效显示处理后的视频流。
*   **JNI (Java Native Interface):** Java/Kotlin 代码与 C++ 代码通过 JNI 进行交互，以执行高性能的 NCNN 推理。
*   **资源管理:** 包含对 `AssetManager` 的使用，以便本地 C++ 代码可以加载存放在 `assets` 目录下的模型文件（如 `.param` 和 `.bin` 文件）。
