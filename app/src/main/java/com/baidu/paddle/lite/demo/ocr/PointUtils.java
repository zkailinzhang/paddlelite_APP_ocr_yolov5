package com.baidu.paddle.lite.demo.ocr;

import android.graphics.Point;
import android.util.Log;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * @ClassName: PointUtils
 * @Description: java类作用描述
 * @Author: GSY
 * @Date: 2021/11/5 16:13
 */
public class PointUtils {
    public static void sort(List<Point> pointList, final Point centetPoint) {
        if (pointList == null || centetPoint == null) return;
        Collections.sort(pointList, new Comparator<Point>() {
            public int compare(Point e1, Point e2) {
                double distance1 = distence(e1, centetPoint);
                double distance2 = distence(e2, centetPoint);
                if (distance1 > distance2) {
                    return 1;
                } else {
                    return -1;
                }
            }
        });
    }

    public static double distence(Point point1, Point point2) {
        return Math.sqrt(Math.pow(point1.x - point2.x, 2) + Math.pow(point1.y - point2.y, 2));
    }

}


