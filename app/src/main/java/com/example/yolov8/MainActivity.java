// package com.example.yolov8; // Make sure this matches your package name
package com.example.yolov8;

import android.content.res.AssetManager;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.nio.ByteBuffer;

// CORRECTED IMPORTS based on official MSDK v5 samples
import dji.sdk.keyvalue.value.common.ComponentIndexType;
import dji.v5.manager.datacenter.MediaDataCenter;
import dji.v5.manager.interfaces.ICameraStreamManager;

public class MainActivity extends AppCompatActivity implements SurfaceHolder.Callback {

    private static final String TAG = "MainActivity_Yolo";

    private SurfaceView surfaceView;
    private Button buttonStartStream;

    private long nativePointer = 0;

    private MediaCodec mediaCodec;
    private MediaCodec.BufferInfo bufferInfo;
    private ICameraStreamManager.ReceiveStreamListener receiveStreamListener;

    private int videoWidth;
    private int videoHeight;

    private Thread decodeThread;
    private volatile boolean isProcessingRunning = false;
    private volatile boolean isDJIStreamDecoderInitialized = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        surfaceView = findViewById(R.id.surfaceView);
        surfaceView.getHolder().addCallback(this);

        buttonStartStream = findViewById(R.id.button_start_stream);
        buttonStartStream.setOnClickListener(v -> toggleStreamProcessing());

        initDJIStreamListener();

        System.loadLibrary("yolov8ncnn");
    }

    private void toggleStreamProcessing() {
        if (isProcessingRunning) {
            stopDroneVideoProcessing();
        } else {
            startDroneVideoProcessing();
        }
    }

    /**
     * Initializes the DJI video stream listener to receive H264 video data.
     */
    private void initDJIStreamListener() {
        // This is the callback that receives raw H264 data from the drone.
        receiveStreamListener = (data, offset, length, info) -> {
            if (mediaCodec != null && isDJIStreamDecoderInitialized) {
                try {
                    int inIndex = mediaCodec.dequeueInputBuffer(10000); // 10ms timeout
                    if (inIndex >= 0) {
                        ByteBuffer buffer = mediaCodec.getInputBuffer(inIndex);
                        if (buffer != null) {
                            buffer.put(data, offset, length);
                            mediaCodec.queueInputBuffer(inIndex, 0, length, System.nanoTime() / 1000, 0);
                        }
                    }
                } catch (IllegalStateException e) {
                    Log.e(TAG, "MediaCodec is in an illegal state, possibly stopped.", e);
                }
            }
        };
    }

    /**
     * Starts receiving and processing the video stream from the drone.
     */
    private void startDroneVideoProcessing() {
        if (isProcessingRunning) return;
        if (nativePointer == 0L) {
            Toast.makeText(this, "Native processor not initialized", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            // 1. Configure MediaCodec for H.264 decoding
            videoWidth = 1920; // Default resolution, can be updated on the fly
            videoHeight = 1080;
            MediaFormat format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, videoWidth, videoHeight);
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar);

            mediaCodec = MediaCodec.createDecoderByType(MediaFormat.MIMETYPE_VIDEO_AVC);
            mediaCodec.configure(format, null, null, 0); // We are not rendering to a surface directly
            mediaCodec.start();
            isDJIStreamDecoderInitialized = true;

            // 2. Get the CameraStreamManager and register the listener
            // THIS IS THE CORRECTED WAY TO GET THE MANAGER
            ICameraStreamManager cameraStreamManager = MediaDataCenter.getInstance().getCameraStreamManager();
            if (cameraStreamManager != null) {
                // Register the listener for the desired camera. FPV is a common choice.
                // THIS IS THE CORRECTED WAY TO SPECIFY THE SOURCE
                cameraStreamManager.addReceiveStreamListener(ComponentIndexType.FPV, receiveStreamListener);

            } else {
                Toast.makeText(this, "CameraStreamManager is not available!", Toast.LENGTH_SHORT).show();
                return;
            }

            // 3. Start the decoding thread
            isProcessingRunning = true;
            decodeThread = new Thread(this::decodeDroneStreamLoop);
            decodeThread.setName("DroneDecodeThread");
            decodeThread.start();

            runOnUiThread(() -> {
                buttonStartStream.setText("停止推流");
                Toast.makeText(MainActivity.this, "Drone video stream started.", Toast.LENGTH_SHORT).show();
            });

        } catch (IOException e) {
            Log.e(TAG, "Error starting drone video processing", e);
            isProcessingRunning = false;
        }
    }

    /**
     * The main loop for the decoding thread.
     */
    private void decodeDroneStreamLoop() {
        bufferInfo = new MediaCodec.BufferInfo();
        while (isProcessingRunning && !Thread.currentThread().isInterrupted()) {
            try {
                int outIndex = mediaCodec.dequeueOutputBuffer(bufferInfo, 10000);

                if (outIndex >= 0) {
                    if (bufferInfo.size > 0) {
                        ByteBuffer outputBuffer = mediaCodec.getOutputBuffer(outIndex);
                        byte[] frameData = new byte[bufferInfo.size];
                        if (outputBuffer != null) {
                            outputBuffer.get(frameData);
                            if (nativePointer != 0L) {
                                // Pass the decoded YUV frame to your JNI/C++ code
                                processFrameNative(nativePointer, frameData, videoWidth, videoHeight,
                                        mediaCodec.getOutputFormat().getInteger(MediaFormat.KEY_COLOR_FORMAT),
                                        bufferInfo.presentationTimeUs);
                            }
                        }
                    }
                    mediaCodec.releaseOutputBuffer(outIndex, false);
                } else if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    MediaFormat newFormat = mediaCodec.getOutputFormat();
                    videoWidth = newFormat.getInteger(MediaFormat.KEY_WIDTH);
                    videoHeight = newFormat.getInteger(MediaFormat.KEY_HEIGHT);
                    Log.d(TAG, "Decoder output format changed to: " + videoWidth + "x" + videoHeight);
                }
            } catch (Exception e) {
                Log.e(TAG, "Exception in decoding loop", e);
                break;
            }
        }
        Log.d(TAG, "Decoding thread finished.");
    }

    /**
     * Stops the video stream processing.
     */
    private void stopDroneVideoProcessing() {
        if (!isProcessingRunning) return;
        isProcessingRunning = false;
        Log.d(TAG, "Attempting to stop drone video processing...");

        if (decodeThread != null) {
            decodeThread.interrupt();
        }

        ICameraStreamManager cameraStreamManager = MediaDataCenter.getInstance().getCameraStreamManager();
        if (cameraStreamManager != null && receiveStreamListener != null) {
            cameraStreamManager.removeReceiveStreamListener(receiveStreamListener);
        }

        if (mediaCodec != null) {
            try {
                mediaCodec.stop();
                mediaCodec.release();
            } catch (Exception e) {
                Log.e(TAG, "Error stopping MediaCodec", e);
            }
            mediaCodec = null;
        }
        isDJIStreamDecoderInitialized = false;

        runOnUiThread(() -> {
            buttonStartStream.setText("开始推流");
            Toast.makeText(MainActivity.this, "Video stream stopped.", Toast.LENGTH_SHORT).show();
            Log.d(TAG, "Drone video processing has been stopped.");
        });
    }

    // --- SurfaceHolder.Callback Methods ---

    @Override
    public void surfaceCreated(@NonNull SurfaceHolder holder) {
        Log.d(TAG, "Surface created");
        initializeNativeProcessor(holder.getSurface());
    }

    @Override
    public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int width, int height) {
        Log.d(TAG, "Surface changed: " + width + "x" + height);
    }

    @Override
    public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
        Log.d(TAG, "Surface destroyed");
        stopDroneVideoProcessing();
        if (nativePointer != 0L) {
            releaseNative(nativePointer);
            nativePointer = 0L;
        }
    }

    // --- Activity Lifecycle Methods ---

    @Override
    protected void onPause() {
        super.onPause();
        stopDroneVideoProcessing();
    }

    // --- JNI Methods ---

    private void initializeNativeProcessor(Surface surface) {
        if (nativePointer != 0) {
            releaseNative(nativePointer);
        }
        nativePointer = initNative(surface, getApplicationContext().getAssets());
        if (nativePointer == 0L) {
            Log.e(TAG, "Failed to initialize native processor!");
            Toast.makeText(this, "Failed to initialize native processor", Toast.LENGTH_LONG).show();
        } else {
            Log.d(TAG, "Native processor initialized successfully.");
        }
    }

    private native long initNative(Surface surface, AssetManager assetManager);
    private native void processFrameNative(long nativePtr, byte[] frameData, int width, int height, int colorFormat, long timestamp);
    private native void releaseNative(long nativePtr);
}