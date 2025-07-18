package com.example.yolov8; // 替换为你的包名

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.Button;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity implements SurfaceHolder.Callback {

    private static final String TAG = "MainActivity_Yolo";

    private SurfaceView surfaceView;
    private Button buttonSelectVideo;

    private VideoProcessor videoProcessor;

    // --- 新增代码 ---
    // 用于存储在 surface 创建之前选择的视频 URI
    private Uri pendingVideoUri = null;
    // --- 新增代码结束 ---

    static {
        System.loadLibrary("yolov8ncnn");
    }

    private final ActivityResultLauncher<Intent> videoPickerLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == Activity.RESULT_OK) {
                    Intent data = result.getData();
                    if (data != null && data.getData() != null) {
                        Uri videoUri = data.getData();
                        Log.d(TAG, "选定的视频 URI: " + videoUri.toString());

                        // --- 修改后的逻辑 ---
                        // 1. 先将选择的 videoUri 存储起来
                        pendingVideoUri = videoUri;

                        // 2. 如果 videoProcessor 已经准备好了，就立即开始处理
                        //    (这种情况发生在用户选择第二个或之后的视频时)
                        if (videoProcessor != null) {
                            videoProcessor.startProcessing(pendingVideoUri, this);
                            // 处理后清除，避免 surface重建时重复播放
                            pendingVideoUri = null;
                        } else {
                            // 如果 videoProcessor 还没准备好，我们什么都不做，
                            // 等待 surfaceCreated 回调来处理 pendingVideoUri
                            Log.d(TAG, "处理器尚未就绪，视频已暂存，等待 Surface 创建。");
                            Toast.makeText(this, "正在准备播放器...", Toast.LENGTH_SHORT).show();
                        }
                        // --- 修改后的逻辑结束 ---

                    } else {
                        Toast.makeText(this, "获取视频URI失败", Toast.LENGTH_SHORT).show();
                    }
                } else {
                    Toast.makeText(this, "视频选择已取消", Toast.LENGTH_SHORT).show();
                }
            });


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        surfaceView = findViewById(R.id.surfaceView);
        surfaceView.getHolder().addCallback(this);

        buttonSelectVideo = findViewById(R.id.button_select_video);
        buttonSelectVideo.setOnClickListener(v -> openFileSelector());
    }

    private void openFileSelector() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("video/*");
        videoPickerLauncher.launch(intent);
    }


    @Override
    public void surfaceCreated(@NonNull SurfaceHolder holder) {
        Log.d(TAG, "Surface 已创建");
        if (videoProcessor == null) {
            videoProcessor = new VideoProcessor(holder.getSurface(), getApplicationContext().getAssets());
        }

        // --- 新增代码 ---
        // 在 Surface 创建好后，检查是否有等待处理的视频
        if (pendingVideoUri != null) {
            Log.d(TAG, "检测到待处理的视频，现在开始播放。");
            videoProcessor.startProcessing(pendingVideoUri, this);
            // 处理完成后，将其置空，防止重复处理
            pendingVideoUri = null;
        }
        // --- 新增代码结束 ---
    }

    @Override
    public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int width, int height) {
        Log.d(TAG, "Surface 尺寸已更改");
    }

    @Override
    public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
        Log.d(TAG, "Surface 已销毁");
        if (videoProcessor != null) {
            videoProcessor.release();
            videoProcessor = null;
        }
        // 当 surface 销毁时，也应该清除待处理的 URI
        pendingVideoUri = null;
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (videoProcessor != null) {
            videoProcessor.stopProcessing();
        }
        Log.d(TAG, "onPause");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (videoProcessor != null) {
            videoProcessor.release();
            videoProcessor = null;
        }
        Log.d(TAG, "onDestroy");
    }
}