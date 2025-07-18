package com.example.yolov8; // 替换为你的包名

import android.content.ContentResolver;
import android.content.Context;
import android.content.res.AssetManager;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import android.view.Surface;
import android.widget.Toast;

import java.io.IOException;
import java.nio.ByteBuffer;

public class VideoProcessor {

    private static final String TAG = "VideoProcessor_Yolo";

    private long nativePointer = 0;
    private volatile boolean isProcessingRunning = false;
    private Thread decodeThread;

    private ParcelFileDescriptor currentVideoFileDescriptor;

    /**
     * 构造函数，初始化原生处理器。
     * @param surface 用于渲染视频帧的 Surface。
     * @param assetManager 用于从 assets 目录加载模型。
     */
    public VideoProcessor(Surface surface, AssetManager assetManager) {
        this.nativePointer = initNative(surface, assetManager);
        if (this.nativePointer == 0L) {
            Log.e(TAG, "关键错误：原生处理器初始化失败！");
            // 在实际应用中，这里可能需要向上层抛出异常
        } else {
            Log.d(TAG, "原生处理器初始化成功。");
        }
    }

    /**
     * 开始处理指定的视频 URI。
     * 此方法会启动一个新的后台线程来解码和处理视频。
     * @param videoUri 要处理的视频的 content URI。
     * @param context 用于获取 ContentResolver 和显示 Toast。
     */
    public void startProcessing(Uri videoUri, Context context) {
        if (isProcessingRunning) {
            Log.w(TAG, "处理已经在运行，请先停止。");
            return;
        }
        if (nativePointer == 0L) {
            Log.e(TAG, "无法开始处理：原生处理器未初始化。");
            Toast.makeText(context, "错误: 原生处理器未就绪", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            // 在开始新的处理前，关闭任何之前打开的描述符
            closeFileDescriptor();
            ContentResolver resolver = context.getContentResolver();
            currentVideoFileDescriptor = resolver.openFileDescriptor(videoUri, "r");
        } catch (IOException e) {
            Log.e(TAG, "打开视频文件描述符失败: " + e.getMessage());
            Toast.makeText(context, "打开视频失败: " + e.getMessage(), Toast.LENGTH_LONG).show();
            return;
        }

        isProcessingRunning = true;
        decodeThread = new Thread(() -> runDecodingLoop(context.getApplicationContext()));
        decodeThread.setName("VideoDecodeThread");
        decodeThread.start();
    }

    /**
     * 停止视频处理线程。
     */
    public void stopProcessing() {
        if (!isProcessingRunning) {
            return;
        }
        Log.d(TAG, "正在尝试停止视频处理...");
        isProcessingRunning = false; // 向线程发送停止信号
        if (decodeThread != null) {
            decodeThread.interrupt(); // 中断线程以唤醒等待
            try {
                decodeThread.join(1000); // 等待线程终止
                if (decodeThread.isAlive()) {
                    Log.w(TAG, "解码线程在超时后仍未终止。");
                }
            } catch (InterruptedException e) {
                Log.w(TAG, "在等待解码线程结束时被中断。", e);
                Thread.currentThread().interrupt();
            }
            decodeThread = null;
        }
        Log.d(TAG, "视频处理已停止。");
    }

    /**
     * 释放所有资源，包括停止线程和释放原生处理器。
     * 这个方法应该在不再需要此类实例时（例如在 surfaceDestroyed 中）调用。
     */
    public void release() {
        stopProcessing();
        closeFileDescriptor();
        if (nativePointer != 0L) {
            releaseNative(nativePointer);
            nativePointer = 0L;
            Log.d(TAG, "原生处理器已释放。");
        }
    }

    private void closeFileDescriptor() {
        if (currentVideoFileDescriptor != null) {
            try {
                currentVideoFileDescriptor.close();
            } catch (IOException e) {
                Log.e(TAG, "关闭 ParcelFileDescriptor 时出错: " + e.getMessage());
            }
            currentVideoFileDescriptor = null;
        }
    }

    /**
     * 视频解码和处理的核心循环，在后台线程中运行。
     */
    private void runDecodingLoop(Context context) {
        MediaExtractor extractor = new MediaExtractor();
        MediaCodec codec = null;

        try {
            extractor.setDataSource(currentVideoFileDescriptor.getFileDescriptor());

            int videoTrackIndex = -1;
            MediaFormat format = null;
            for (int i = 0; i < extractor.getTrackCount(); i++) {
                MediaFormat trackFormat = extractor.getTrackFormat(i);
                String mime = trackFormat.getString(MediaFormat.KEY_MIME);
                if (mime != null && mime.startsWith("video/")) {
                    videoTrackIndex = i;
                    format = trackFormat;
                    extractor.selectTrack(videoTrackIndex);
                    codec = MediaCodec.createDecoderByType(mime);
                    break;
                }
            }

            if (videoTrackIndex == -1 || format == null || codec == null) {
                Log.e(TAG, "未找到视频轨道或创建解码器失败。");
                return;
            }

            int videoWidth = format.getInteger(MediaFormat.KEY_WIDTH);
            int videoHeight = format.getInteger(MediaFormat.KEY_HEIGHT);

            // 配置解码器，注意这里的 Surface 是 null，因为我们想手动获取解码后的数据
            codec.configure(format, null, null, 0);
            codec.start();

            MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
            boolean isInputEOS = false;
            boolean isOutputEOS = false;

            while (isProcessingRunning && !isOutputEOS && !Thread.currentThread().isInterrupted()) {
                // 将数据送入解码器
                if (!isInputEOS) {
                    int inputBufferId = codec.dequeueInputBuffer(10000);
                    if (inputBufferId >= 0) {
                        ByteBuffer inputBuffer = codec.getInputBuffer(inputBufferId);
                        int sampleSize = extractor.readSampleData(inputBuffer, 0);
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(inputBufferId, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            isInputEOS = true;
                        } else {
                            codec.queueInputBuffer(inputBufferId, 0, sampleSize, extractor.getSampleTime(), 0);
                            extractor.advance();
                        }
                    }
                }

                // 从解码器获取数据
                int outputBufferId = codec.dequeueOutputBuffer(bufferInfo, 10000);
                if (outputBufferId >= 0) {
                    ByteBuffer outputBuffer = codec.getOutputBuffer(outputBufferId);
                    if (outputBuffer != null && bufferInfo.size > 0 && nativePointer != 0L) {
                        byte[] frameData = new byte[bufferInfo.size];
                        outputBuffer.get(frameData);

                        // 将解码后的帧数据传递给原生代码进行处理和渲染
                        processFrameNative(nativePointer, frameData, videoWidth, videoHeight,
                                codec.getOutputFormat().getInteger(MediaFormat.KEY_COLOR_FORMAT),
                                bufferInfo.presentationTimeUs);
                    }
                    // 释放 buffer 以便解码器可以重用它。
                    // 注意：当使用 Surface 进行渲染时，第二个参数为 true。
                    // 由于我们在这里不直接渲染到 MediaCodec 的 surface，而是将数据传递给 C++，
                    // C++ 中的 ANativeWindow_lock 和 unlockAndPost 会处理渲染，
                    // 因此这里 releaseOutputBuffer 的 render 参数应为 false。
                    codec.releaseOutputBuffer(outputBufferId, false);

                    if ((bufferInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        isOutputEOS = true;
                    }
                } else if (outputBufferId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    Log.d(TAG, "解码器输出格式已更改: " + codec.getOutputFormat());
                    // 如果需要，可以更新宽度/高度等信息
                    videoWidth = codec.getOutputFormat().getInteger(MediaFormat.KEY_WIDTH);
                    videoHeight = codec.getOutputFormat().getInteger(MediaFormat.KEY_HEIGHT);
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "视频处理期间出现IO错误: ", e);
        } catch (IllegalStateException e) {
            Log.e(TAG, "视频处理期间出现状态错误 (通常是编解码器问题): ", e);
        } catch (Exception e) {
            Log.e(TAG, "视频处理期间出现未知错误: ", e);
        } finally {
            try {
                if (codec != null) {
                    codec.stop();
                    codec.release();
                }
                extractor.release();
            } catch (Exception e) {
                Log.e(TAG, "释放编解码器/提取器资源时出错", e);
            }
            Log.d(TAG, "视频处理线程结束。");
            isProcessingRunning = false;
        }
    }

    // --- JNI 方法 ---
    // 这些方法现在是 VideoProcessor 类的一部分
    private native long initNative(Surface surface, AssetManager assetManager);
    private native void processFrameNative(long nativePtr, byte[] frameData, int width, int height, int colorFormat, long timestamp);
    private native void releaseNative(long nativePtr);
}