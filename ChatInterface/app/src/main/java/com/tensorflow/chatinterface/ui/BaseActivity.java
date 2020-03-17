package com.tensorflow.chatinterface.ui;

import android.annotation.TargetApi;
import android.app.Activity;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.ViewGroup;
import android.widget.LinearLayout;

import com.tensorflow.chatinterface.R;


public class BaseActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setTranslucent();
    }

    /**
     * 设置沉浸式状态栏
     */
    @TargetApi(Build.VERSION_CODES.LOLLIPOP)
    private void setTranslucent() {
        ViewGroup decorView = (ViewGroup) getWindow().getDecorView();
        int count = decorView.getChildCount();
        //判断是否已经添加了statusBarView
        if (count > 0 && decorView.getChildAt(count - 1) instanceof StatusBarView)
            decorView.getChildAt(count - 1).setBackground(getDrawable(R.drawable.tpv_window_background));
        else {
            //新建一个和状态栏高宽的view
            getWindow().setStatusBarColor(Color.TRANSPARENT);
            StatusBarView statusView = createStatusBarView(this, 0xff3F51B5, 0);
            statusView.setBackground(getDrawable(R.drawable.tpv_window_background));
            decorView.addView(statusView);
        }
        ViewGroup rootView = (ViewGroup) ((ViewGroup) findViewById(android.R.id.content)).getChildAt(0);
        //rootview不会为状态栏留出状态栏空间
        rootView.setFitsSystemWindows(true);
        rootView.setClipToPadding(true);
    }

    /**
     * 创建一个状态栏控件
     *
     * @param activity
     * @param color
     * @param alpha
     * @return
     */
    private StatusBarView createStatusBarView(Activity activity, int color, int alpha) {
        // 绘制一个和状态栏一样高的矩形
        StatusBarView statusBarView = new StatusBarView(activity);
        int identifier = getResources().getIdentifier("status_bar_height", "dimen", "android");
        int height = -1;
        if (identifier > 0) {
            height = getResources().getDimensionPixelSize(identifier);
        }
        LinearLayout.LayoutParams params =
                new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, height);
        statusBarView.setLayoutParams(params);
        statusBarView.setBackgroundColor(color);
        return statusBarView;
    }
}
