// package com.example.yolov8; // 请确保这里的包名和您项目的一致
package com.example.yolov8;

import android.app.Application;
import android.content.Context;
import android.util.Log;

import dji.v5.common.error.IDJIError;
import dji.v5.common.register.DJISDKInitEvent;
import dji.v5.manager.SDKManager;
import dji.v5.manager.interfaces.SDKManagerCallback;

public class MyApplication extends Application {

    private static final String TAG = "MyApplication_Yolo";

    @Override
    protected void attachBaseContext(Context base) {
        super.attachBaseContext(base);
        // MSDK V5 需要在此进行初始化
        SDKManager.getInstance().init(this, new SDKManagerCallback() {
            @Override
            public void onInitProcess(DJISDKInitEvent event, int totalProcess) {
                // --- 这里是修正后代码 ---
                // 'event' 对象本身就是状态枚举，我们直接调用 .name() 获取其字符串名称
                // 移除了错误的 .getStage() 调用
                Log.i(TAG, "onInitProcess: " + event.name());

                // 这部分逻辑是正确的，保持不变
                if (event == DJISDKInitEvent.INITIALIZE_COMPLETE) {
                    SDKManager.getInstance().registerApp();
                }
            }

            @Override
            public void onRegisterSuccess() {
                Log.i(TAG, "onRegisterSuccess: MSDK 注册成功");
            }

            @Override
            public void onRegisterFailure(IDJIError error) {
                Log.e(TAG, "onRegisterFailure: " + error.description());
            }

            @Override
            public void onProductConnect(int productId) {
                Log.i(TAG, "onProductConnect: " + productId);
            }

            @Override
            public void onProductDisconnect(int productId) {
                Log.i(TAG, "onProductDisconnect: " + productId);
            }

            @Override
            public void onProductChanged(int productId) {
                Log.i(TAG, "onProductChanged: " + productId);
            }

            @Override
            public void onDatabaseDownloadProgress(long current, long total) {
                // 本示例中未使用
            }
        });
    }
}