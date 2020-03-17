package com.tensorflow.chatinterface.util;

import android.content.Context;
import android.util.TypedValue;

/**
 * Author:liuyanguo
 * Date:2018/8/2
 * Time:11:30
 * Description:获取手机固定信息工具类
 */
public class SystemUtils {

    /**
     * 获取手机状态栏高度
     *
     * @param context
     * @return
     */
    public static int getStatusBarHeight(Context context) {
        int result = 0;
        int resourceId = context.getResources().getIdentifier(
                "status_bar_height", "dimen", "android");
        if (resourceId > 0) {
            result = context.getResources().getDimensionPixelSize(resourceId);
        }
        return result;
    }

    /**
     * 获取手机ActionBar高度
     *
     * @param context
     * @return
     */
    public static int getActionBarHeight(Context context) {
        int actionBarHeight = 0;
        TypedValue tv = new TypedValue();
        if (context.getTheme().resolveAttribute(android.R.attr.actionBarSize, tv, true)) {
            actionBarHeight = TypedValue.complexToDimensionPixelSize(
                    tv.data, context.getResources().getDisplayMetrics());
        }
        return actionBarHeight;
    }
}
