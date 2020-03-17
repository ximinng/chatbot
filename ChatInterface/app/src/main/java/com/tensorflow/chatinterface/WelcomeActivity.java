package com.tensorflow.chatinterface;

import android.content.Intent;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;

import com.tensorflow.chatinterface.ui.ChatActivity;

public class WelcomeActivity extends AppCompatActivity implements Runnable{

    final Handler mHandler = new Handler();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_welcome);

        mHandler.postDelayed(this, 2000);
    }

    @Override
    public void run() {
        Intent intent = new Intent(this, ChatActivity.class);
        startActivity(intent);
        // 此处可以不需要调用finish()了, 因为已经设置了noHistory属性, 从而使得系统接管finish操作
    }
}
