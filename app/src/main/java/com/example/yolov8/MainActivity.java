package com.example.yolov8; // 替换为你的包名

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import java.io.IOException;
import android.os.Build; // 确保导入这个类

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int REQUEST_CODE_PERMISSION = 101;

    private ImageView imageView;
    private TextView textViewFPS;
    private Button buttonSelect;
    private ActivityResultLauncher<Intent> videoPickerLauncher;
    private ExecutorService executorService;
    private MediaMetadataRetriever mediaMetadataRetriever;
    private volatile boolean isProcessing = false;

    // 加载我们创建的 JNI 库
    static {
        System.loadLibrary("yolov8ncnn");
    }

    // 定义 native 方法ƒ
    public native boolean initYolo(AssetManager assetManager);
    public native boolean detect(Bitmap bitmap);


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 检查权限
        checkPermissions();

        // 初始化视图
        imageView = findViewById(R.id.imageView);
        textViewFPS = findViewById(R.id.textViewFPS);
        buttonSelect = findViewById(R.id.buttonSelect);

        // 初始化后台线程池
        executorService = Executors.newSingleThreadExecutor();

        // 初始化模型
        executorService.execute(() -> {
            boolean success = initYolo(getAssets());
            if (!success) {
                Log.e(TAG, "Failed to initialize YOLO model.");
            } else {
                Log.i(TAG, "YOLO model initialized successfully.");
            }
        });

        // 设置视频选择器
        videoPickerLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Uri videoUri = result.getData().getData();
                        if (videoUri != null) {
                            startVideoProcessing(videoUri);
                        }
                    }
                }
        );

        buttonSelect.setOnClickListener(v -> {
            if (isProcessing) {
                stopVideoProcessing();
            } else {
                Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("video/*");
                videoPickerLauncher.launch(intent);
            }
        });
    }

    private void startVideoProcessing(Uri videoUri) {
        isProcessing = true;
        buttonSelect.setText("停止处理");
        mediaMetadataRetriever = new MediaMetadataRetriever();
        try {
            mediaMetadataRetriever.setDataSource(this, videoUri);
        } catch (Exception e) {
            Log.e(TAG, "Error setting data source for MediaMetadataRetriever", e);
            Toast.makeText(this, "无法打开视频文件", Toast.LENGTH_SHORT).show();
            stopVideoProcessing();
            return;
        }

        executorService.execute(() -> {
            long startTime, endTime;
            String durationStr = mediaMetadataRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            long durationMs = Long.parseLong(durationStr);

            // 每秒处理 10 帧 (100ms per frame)
            for (long timeMs = 0; timeMs < durationMs && isProcessing; timeMs += 100) {
                startTime = System.currentTimeMillis();

                // 获取视频帧
                final Bitmap bitmap = mediaMetadataRetriever.getFrameAtTime(timeMs * 1000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
                if (bitmap == null) continue;

                Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);

                // 调用 JNI 进行检测
                detect(mutableBitmap);

                endTime = System.currentTimeMillis();

                // ✅ 为 Lambda 创建一个 "final" 的变量副本
                // 这个副本只在当前这次循环中有效，并且它的值是固定的
                final long finalStartTime = startTime;
                final long finalEndTime = endTime;

                runOnUiThread(() -> {
                    imageView.setImageBitmap(mutableBitmap);

                    // ✅ 在 Lambda 中使用这个 final 的副本
                    long fps = (finalEndTime - finalStartTime) > 0 ? 1000 / (finalEndTime - finalStartTime) : 0;
                    textViewFPS.setText(String.format("FPS: %d", fps));
                });
            }
            // 处理完成
            runOnUiThread(this::stopVideoProcessing);
        });
    }

    private void stopVideoProcessing() {
        isProcessing = false;
        if (mediaMetadataRetriever != null) {
            try {
                // 我们“尝试”执行这个可能会出问题的操作
                mediaMetadataRetriever.release();
            } catch (IOException e) {
                // 如果真的发生了 IOException，我们就在这里“捕获”它
                // 最佳实践是打印错误日志，方便调试
                Log.e(TAG, "Error releasing MediaMetadataRetriever", e);
                e.printStackTrace();
            } finally {
                // finally 代码块里的内容，无论是否发生异常，都一定会被执行
                // 确保 mediaMetadataRetriever 变量被设为 null
                mediaMetadataRetriever = null;
            }
        }
        buttonSelect.setText("选择视频");
        // 确保UI操作在主线程执行
        runOnUiThread(() -> {
            Toast.makeText(this, "处理已停止", Toast.LENGTH_SHORT).show();
        });
    }
    private void checkPermissions() {
        String permission;
        // 检查安卓版本
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) { // TIRAMISU 是 Android 13 (API 33)
            // Android 13及以上版本，请求新的视频权限
            permission = Manifest.permission.READ_MEDIA_VIDEO;
        } else {
            // Android 12及以下版本，请求旧的存储权限
            permission = Manifest.permission.READ_EXTERNAL_STORAGE;
        }

        // 检查权限是否已经被授予
        if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
            // 如果没有被授予，则请求权限
            ActivityCompat.requestPermissions(this, new String[]{permission}, REQUEST_CODE_PERMISSION);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSION) {
            if (grantResults.length <= 0 || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "需要存储权限才能选择视频", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopVideoProcessing();
        executorService.shutdown();
    }
}