package com.txooo.cameraface.utils;

import java.text.DecimalFormat;

/**
 * Created by admin on 2017/12/13.
 */

public class NumFormater {


    /**
     * double 数据 转百分比
     *
     * @param decimalNum
     * @return
     */
    public static String numDecFormat(double decimalNum) {
        DecimalFormat df = new DecimalFormat("0.00%");
        return df.format(decimalNum);
    }

}
