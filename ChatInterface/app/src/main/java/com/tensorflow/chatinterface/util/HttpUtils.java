package com.tensorflow.chatinterface.util;

import android.text.TextUtils;
import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;


public class HttpUtils {
    private static final String TAG = "HttpUtils";
    public static final MediaType JSON = MediaType.parse("application/json; charset=utf-8");
    private static OkHttpClient mClient = null;
    private static HttpUtils sInstance = null;

    private HttpUtils() {
    }

    public static HttpUtils getInstance() {
        if (null == sInstance) {
            synchronized (HttpUtils.class) {
                if (null == sInstance) {
                    sInstance = new HttpUtils();
                }
            }
        }
        return sInstance;
    }

    /**
     * 通用同步请求
     *
     * @param request
     * @return
     * @throws IOException
     */
    private Response execute(Request request) {
        mClient = new OkHttpClient.Builder()
                .connectTimeout(1, TimeUnit.MINUTES)
                .writeTimeout(5, TimeUnit.MINUTES)
                .readTimeout(5, TimeUnit.MINUTES)
                .build();
        try {
            return mClient.newCall(request).execute();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * get方式URL拼接
     *
     * @param url
     * @param message
     * @return
     */
    public String getRequestUrl(String url, String message) {
        if (TextUtils.isEmpty(message)) {
            return url;
        }
        StringBuilder newUrl = new StringBuilder(url);
//        newUrl.append("?");
        newUrl.append("info");
//        newUrl.append("infos");
        newUrl.append("=");
        newUrl.append(message);

//        Log.d(TAG," ------URL-------: " + newUrl);
        return newUrl.toString();
    }

    /**
     * get请求获取response
     *
     * @param url
     * @return
     */
    public String getRequest(String url) {
        String responseString = null;
        Request request = new Request.Builder()
                .url(url)
                .build();
        try {
            Response response = execute(request);
            responseString = response.body().string();
        } catch (IOException e) {
            e.printStackTrace();
        }
//        JsonParser.parseGrammarResult(responseString);

        JSONObject jsonObj = null;
        try {
            jsonObj = new JSONObject(responseString);
            String text = (String) jsonObj.get("text");
            Log.d(TAG, "responseString = " + text);
            return text;
        } catch (JSONException e) {
            e.printStackTrace();
        }

//        Log.d(TAG,"responseString = " + responseString);
        return new String("解析JSON错误");
    }
}
