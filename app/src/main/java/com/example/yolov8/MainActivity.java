package com.example.yolov8; // 替换为你的包名

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.FileDescriptor;
import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity implements SurfaceHolder.Callback {

    private static final String TAG = "MainActivity_Yolo";
    private static final int PERMISSION_REQUEST_CODE_READ_STORAGE = 101;
    private static final int SELECT_VIDEO_REQUEST_CODE = 42; // Or use ActivityResultLauncher

    private SurfaceView surfaceView;
    private SurfaceHolder surfaceHolder;
    private Button buttonSelectVideo;
    private long nativePointer = 0;

    private String currentVideoPath; // Can be a content URI path or a direct file path
    private ParcelFileDescriptor currentVideoFileDescriptor; // For content URIs

    private MediaExtractor mediaExtractor;
    private MediaCodec mediaCodec;
    private MediaCodec.BufferInfo bufferInfo;

    private int videoWidth;
    private int videoHeight;

    private Thread decodeThread;
    private volatile boolean isProcessingRunning = false;


    // ActivityResultLauncher for picking a video
    private final ActivityResultLauncher<Intent> videoPickerLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == Activity.RESULT_OK) {
                    Intent data = result.getData();
                    if (data != null && data.getData() != null) {
                        Uri videoUri = data.getData();
                        Log.d(TAG, "Video URI selected: " + videoUri.toString());
                        // Stop any previous processing
                        stopVideoProcessing();
                        // Reset native processor if it was initialized for a different surface/video
                        if (nativePointer != 0) {
                            releaseNative(nativePointer);
                            nativePointer = 0; // Will be re-initialized in surfaceCreated
                        }
                        handleSelectedVideo(videoUri);
                    } else {
                        Toast.makeText(this, "Failed to get video URI", Toast.LENGTH_SHORT).show();
                    }
                } else {
                    Toast.makeText(this, "Video selection cancelled", Toast.LENGTH_SHORT).show();
                }
            });


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        surfaceView = findViewById(R.id.surfaceView);
        surfaceHolder = surfaceView.getHolder();
        surfaceHolder.addCallback(this);

        buttonSelectVideo = findViewById(R.id.button_select_video);
        buttonSelectVideo.setOnClickListener(v -> openFileSelector());

        System.loadLibrary("yolov8ncnn");
    }

    private void openFileSelector() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT); // SAF for modern Android
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("video/*");
        videoPickerLauncher.launch(intent);
    }

    private void handleSelectedVideo(Uri videoUri) {
        try {
            // For content URIs, we need to open a ParcelFileDescriptor
            // We'll pass the FileDescriptor from this to MediaExtractor
            if (currentVideoFileDescriptor != null) {
                try {
                    currentVideoFileDescriptor.close();
                } catch (IOException e) {
                    Log.e(TAG, "Error closing previous PFD: " + e.getMessage());
                }
            }
            currentVideoFileDescriptor = getContentResolver().openFileDescriptor(videoUri, "r");
            currentVideoPath = videoUri.toString(); // Store URI for reference

            // If the surface is already created, we can attempt to start processing.
            // Otherwise, surfaceCreated will handle it.
            if (surfaceHolder.getSurface() != null && surfaceHolder.getSurface().isValid() && nativePointer != 0) {
                startVideoProcessing();
            } else if (surfaceHolder.getSurface() == null) {
                Log.w(TAG,"Surface not ready yet, video processing will start when surface is created.");
            } else if (nativePointer == 0 && surfaceHolder.getSurface() != null && surfaceHolder.getSurface().isValid()){
                Log.w(TAG,"Native processor not ready, re-initializing in surfaceCreated (if not already called).");
                // Potentially re-init native if surface is valid but nativePointer is not
                // This case might be tricky if surfaceCreated was already called
                initializeNativeProcessor(surfaceHolder.getSurface());
                if(nativePointer != 0) startVideoProcessing();

            }

        } catch (IOException e) {
            Log.e(TAG, "Error opening video file descriptor: " + e.getMessage());
            Toast.makeText(this, "Error opening video: " + e.getMessage(), Toast.LENGTH_LONG).show();
            currentVideoFileDescriptor = null;
            currentVideoPath = null;
        }
    }


    private void initializeNativeProcessor(Surface surface) {
        if (nativePointer != 0) {
            releaseNative(nativePointer); // Release previous instance if any
        }
        nativePointer = initNative(surface, getApplicationContext().getAssets());
        if (nativePointer == 0L) {
            Log.e(TAG, "Failed to initialize native processor in initializeNativeProcessor");
            Toast.makeText(this, "Failed to initialize native processor", Toast.LENGTH_LONG).show();
        } else {
            Log.d(TAG, "Native processor initialized successfully via initializeNativeProcessor.");
        }
    }


    @Override
    public void surfaceCreated(@NonNull SurfaceHolder holder) {
        Log.d(TAG, "Surface created");
        initializeNativeProcessor(holder.getSurface());
        // If a video was selected *before* the surface was ready, start processing now.
        if (currentVideoFileDescriptor != null && nativePointer != 0 && !isProcessingRunning) {
            startVideoProcessing();
        }
    }

    @Override
    public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int width, int height) {
        Log.d(TAG, "Surface changed: currently not supported");
    }

    @Override
    public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
        Log.d(TAG, "Surface destroyed");
        stopVideoProcessing();
        if (nativePointer != 0L) {
            releaseNative(nativePointer);
            nativePointer = 0L;
        }
        if (currentVideoFileDescriptor != null) {
            try {
                currentVideoFileDescriptor.close();
            } catch (IOException e) {
                Log.e(TAG, "Error closing PFD on surfaceDestroyed: " + e.getMessage());
            }
            currentVideoFileDescriptor = null;
        }
    }

    private void startVideoProcessing() {
        if (isProcessingRunning) {
            Log.w(TAG, "Video processing is already running.");
            return;
        }
        if (currentVideoFileDescriptor == null || nativePointer == 0L) {
            Log.e(TAG, "Cannot start video processing: Video not selected or native processor not ready.");
            if (currentVideoFileDescriptor == null) Toast.makeText(this, "Please select a video first.", Toast.LENGTH_SHORT).show();
            return;
        }

        isProcessingRunning = true;
        decodeThread = new Thread(() -> {
            MediaExtractor tempExtractor = null; // Use local var inside thread
            MediaCodec tempCodec = null;     // Use local var inside thread
            try {
                tempExtractor = new MediaExtractor();
                // Use FileDescriptor from ParcelFileDescriptor
                tempExtractor.setDataSource(currentVideoFileDescriptor.getFileDescriptor());

                int videoTrackIndex = -1;
                MediaFormat inputFormat = null;
                for (int i = 0; i < tempExtractor.getTrackCount(); i++) {
                    MediaFormat format = tempExtractor.getTrackFormat(i);
                    String mime = format.getString(MediaFormat.KEY_MIME);
                    if (mime != null && mime.startsWith("video/")) {
                        videoTrackIndex = i;
                        videoWidth = format.getInteger(MediaFormat.KEY_WIDTH);
                        videoHeight = format.getInteger(MediaFormat.KEY_HEIGHT);
                        inputFormat = format;
                        tempExtractor.selectTrack(videoTrackIndex);
                        tempCodec = MediaCodec.createDecoderByType(mime);
                        break;
                    }
                }

                if (videoTrackIndex == -1 || inputFormat == null || tempCodec == null) {
                    Log.e(TAG, "No video track found, input format is null, or codec creation failed.");
                    runOnUiThread(()-> Toast.makeText(MainActivity.this, "Failed to setup video decoder.", Toast.LENGTH_LONG).show());
                    return;
                }

                MediaFormat outputFormat = MediaFormat.createVideoFormat(inputFormat.getString(MediaFormat.KEY_MIME), videoWidth, videoHeight);
                outputFormat.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar);
                tempCodec.configure(outputFormat, null, null, 0);
                tempCodec.start();
                bufferInfo = new MediaCodec.BufferInfo(); // Member variable

                Log.d(TAG, "Video processing thread started for: " + currentVideoPath);

                boolean isInputEOS = false;
                boolean isOutputEOS = false;

                while (isProcessingRunning && !isOutputEOS && !Thread.currentThread().isInterrupted()) {
                    if (!isInputEOS) {
                        int inputBufferId = tempCodec.dequeueInputBuffer(10000);
                        if (inputBufferId >= 0) {
                            ByteBuffer inputBuffer = tempCodec.getInputBuffer(inputBufferId);
                            if (inputBuffer != null) {
                                int sampleSize = tempExtractor.readSampleData(inputBuffer, 0);
                                if (sampleSize < 0) {
                                    tempCodec.queueInputBuffer(inputBufferId, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                                    isInputEOS = true;
                                    Log.d(TAG, "Input EOS");
                                } else {
                                    tempCodec.queueInputBuffer(inputBufferId, 0, sampleSize, tempExtractor.getSampleTime(), 0);
                                    tempExtractor.advance();
                                }
                            }
                        }
                    }

                    int outputBufferId = tempCodec.dequeueOutputBuffer(bufferInfo, 10000);
                    if (outputBufferId >= 0) {
                        ByteBuffer outputBuffer = tempCodec.getOutputBuffer(outputBufferId);
                        if (outputBuffer != null && bufferInfo.size > 0 && nativePointer != 0L) {
                            byte[] frameData = new byte[bufferInfo.size];
                            outputBuffer.get(frameData);
                            // Process frame directly to avoid complexities with queue for now
                            processFrameNative(nativePointer, frameData, videoWidth, videoHeight,
                                    tempCodec.getOutputFormat().getInteger(MediaFormat.KEY_COLOR_FORMAT),
                                    bufferInfo.presentationTimeUs);
                        }
                        if (outputBuffer != null) outputBuffer.clear(); // Clear before releasing
                        tempCodec.releaseOutputBuffer(outputBufferId, false);

                        if ((bufferInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                            Log.d(TAG, "Output EOS");
                            isOutputEOS = true;
                        }
                    } else if (outputBufferId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        MediaFormat newFormat = tempCodec.getOutputFormat();
                        Log.d(TAG, "Decoder output format changed: " + newFormat);
                        videoWidth = newFormat.getInteger(MediaFormat.KEY_WIDTH);
                        videoHeight = newFormat.getInteger(MediaFormat.KEY_HEIGHT);
                        // Notify C++ if necessary
                    }
                }

            } catch (IOException e) {
                Log.e(TAG, "Error during video processing IO: ", e);
                runOnUiThread(()-> Toast.makeText(MainActivity.this, "IO Error: " + e.getMessage(), Toast.LENGTH_LONG).show());
            } catch (IllegalStateException e) {
                Log.e(TAG, "Error during video processing State: ", e);
                runOnUiThread(()-> Toast.makeText(MainActivity.this, "Codec Error: " + e.getMessage(), Toast.LENGTH_LONG).show());
            } catch (Exception e) {
                Log.e(TAG, "Generic error during video processing: ", e);
                runOnUiThread(()-> Toast.makeText(MainActivity.this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show());
            }
            finally {
                try {
                    if (tempCodec != null) {
                        tempCodec.stop();
                        tempCodec.release();
                    }
                    if (tempExtractor != null) {
                        tempExtractor.release();
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Error releasing codec/extractor resources", e);
                }
                Log.d(TAG, "Video processing thread finished.");
                isProcessingRunning = false; // Mark as not running
            }
        });
        decodeThread.setName("VideoDecodeThread");
        decodeThread.start();
    }

    private void stopVideoProcessing() {
        Log.d(TAG, "Attempting to stop video processing...");
        isProcessingRunning = false; // Signal thread to stop
        if (decodeThread != null) {
            decodeThread.interrupt();
            try {
                decodeThread.join(1000); // Wait for thread to finish
                if (decodeThread.isAlive()) {
                    Log.w(TAG, "Decode thread did not terminate in time.");
                }
            } catch (InterruptedException e) {
                Log.w(TAG, "Interrupted while waiting for decode thread to finish.", e);
                Thread.currentThread().interrupt();
            }
            decodeThread = null;
        }
        Log.d(TAG,"Video processing stopped.");
    }

    @Override
    protected void onPause() {
        super.onPause();
        // If you want to stop processing when the app is paused
        // stopVideoProcessing();
        // if (nativePointer != 0L) {
        // releaseNative(nativePointer); // Be careful with surface state
        // nativePointer = 0L;
        // }
        Log.d(TAG, "onPause");
    }

    @Override
    protected void onResume() {
        super.onResume();
        // If you stopped processing in onPause and surface is valid,
        // you might want to reinitialize and restart if a video was selected.
        // if (surfaceHolder.getSurface() != null && surfaceHolder.getSurface().isValid()) {
        //     if (nativePointer == 0L) {
        //         initializeNativeProcessor(surfaceHolder.getSurface());
        //     }
        //     if (currentVideoFileDescriptor != null && nativePointer != 0 && !isProcessingRunning) {
        //         startVideoProcessing();
        //     }
        // }
        Log.d(TAG, "onResume");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Ensure everything is cleaned up. SurfaceDestroyed should handle most of this.
        stopVideoProcessing(); // Final stop
        if (nativePointer != 0L) {
            releaseNative(nativePointer);
            nativePointer = 0L;
        }
        if (currentVideoFileDescriptor != null) {
            try {
                currentVideoFileDescriptor.close();
            } catch (IOException e) {
                Log.e(TAG, "Error closing PFD on onDestroy: " + e.getMessage());
            }
            currentVideoFileDescriptor = null;
        }
        Log.d(TAG, "onDestroy");
    }


    // --- JNI 方法 ---
    private native long initNative(Surface surface, AssetManager assetManager);
    private native void processFrameNative(long nativePtr, byte[] frameData, int width, int height, int colorFormat, long timestamp);
    private native void releaseNative(long nativePtr);
}
