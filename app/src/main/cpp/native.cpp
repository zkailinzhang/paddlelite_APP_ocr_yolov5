//
// Created by fujiayi on 2020/7/5.
//

#include "native.h"
#include "ocr_ppredictor.h"
#include <algorithm>
#include <paddle_api.h>
#include <string>


static paddle::lite_api::PowerMode str_to_cpu_mode(const std::string &cpu_mode);

extern "C" JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_demo_ocr_OCRPredictorNative_init(
    JNIEnv *env, jobject thiz, jstring j_det_model_path,
    jstring j_rec_model_path, jstring j_cls_model_path, jint j_thread_num,
    jstring j_cpu_mode) {
  std::string det_model_path = jstring_to_cpp_string(env, j_det_model_path);
  std::string rec_model_path = jstring_to_cpp_string(env, j_rec_model_path);
  std::string cls_model_path = jstring_to_cpp_string(env, j_cls_model_path);
  int thread_num = j_thread_num;
  std::string cpu_mode = jstring_to_cpp_string(env, j_cpu_mode);
  ppredictor::OCR_Config conf;
  conf.thread_num = thread_num;
  conf.mode = str_to_cpu_mode(cpu_mode);
  ppredictor::OCR_PPredictor *orc_predictor =
      new ppredictor::OCR_PPredictor{conf};
  orc_predictor->init_from_file(det_model_path, rec_model_path, cls_model_path);
  return reinterpret_cast<jlong>(orc_predictor);
}

/**
 * "LITE_POWER_HIGH" convert to paddle::lite_api::LITE_POWER_HIGH
 * @param cpu_mode
 * @return
 */
static paddle::lite_api::PowerMode
str_to_cpu_mode(const std::string &cpu_mode) {
  static std::map<std::string, paddle::lite_api::PowerMode> cpu_mode_map{
      {"LITE_POWER_HIGH", paddle::lite_api::LITE_POWER_HIGH},
      {"LITE_POWER_LOW", paddle::lite_api::LITE_POWER_HIGH},
      {"LITE_POWER_FULL", paddle::lite_api::LITE_POWER_FULL},
      {"LITE_POWER_NO_BIND", paddle::lite_api::LITE_POWER_NO_BIND},
      {"LITE_POWER_RAND_HIGH", paddle::lite_api::LITE_POWER_RAND_HIGH},
      {"LITE_POWER_RAND_LOW", paddle::lite_api::LITE_POWER_RAND_LOW}};
  std::string upper_key;
  std::transform(cpu_mode.cbegin(), cpu_mode.cend(), upper_key.begin(),
                 ::toupper);
  auto index = cpu_mode_map.find(upper_key);
  if (index == cpu_mode_map.end()) {
    LOGE("cpu_mode not found %s", upper_key.c_str());
    return paddle::lite_api::LITE_POWER_HIGH;
  } else {
    return index->second;
  }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_baidu_paddle_lite_demo_ocr_OCRPredictorNative_forward(
    JNIEnv *env, jobject thiz, jlong java_pointer, jfloatArray buf,
    jfloatArray ddims, jobject original_image) {
  LOGI("begin to run native forward");
  if (java_pointer == 0) {
    LOGE("JAVA pointer is NULL");
    return cpp_array_to_jfloatarray(env, nullptr, 0);
  }
  cv::Mat origin = bitmap_to_cv_mat(env, original_image);
  if (origin.size == 0) {
    LOGE("origin bitmap cannot convert to CV Mat");
    return cpp_array_to_jfloatarray(env, nullptr, 0);
  }
  ppredictor::OCR_PPredictor *ppredictor =
      (ppredictor::OCR_PPredictor *)java_pointer;
  std::vector<float> dims_float_arr = jfloatarray_to_float_vector(env, ddims);
  std::vector<int64_t> dims_arr;
  dims_arr.resize(dims_float_arr.size());
  std::copy(dims_float_arr.cbegin(), dims_float_arr.cend(), dims_arr.begin());

  // 这里值有点大，就不调用jfloatarray_to_float_vector了
  int64_t buf_len = (int64_t)env->GetArrayLength(buf);
  jfloat *buf_data = env->GetFloatArrayElements(buf, JNI_FALSE);
  float *data = (jfloat *)buf_data;
  std::vector<ppredictor::OCRPredictResult> results =
      ppredictor->infer_ocr(dims_arr, data, buf_len, NET_OCR, origin);
  LOGI("infer_ocr finished with boxes %ld", results.size());
  // 这里将std::vector<ppredictor::OCRPredictResult> 序列化成
  // float数组，传输到java层再反序列化
  std::vector<float> float_arr;
  for (const ppredictor::OCRPredictResult &r : results) {
    float_arr.push_back(r.points.size());
    float_arr.push_back(r.word_index.size());
    float_arr.push_back(r.score);
    for (const std::vector<int> &point : r.points) {
      float_arr.push_back(point.at(0));
      float_arr.push_back(point.at(1));
    }
    for (int index : r.word_index) {
      float_arr.push_back(index);
    }
  }
  return cpp_array_to_jfloatarray(env, float_arr.data(), float_arr.size());
}


//识别函数
//--参数：输入图像(numpy),量程,极坐标中心(行,列),半径
//--返回值：读数score_final
float round_meter(const cv::Mat& image, float start,float end, cv::Point2f center, int r, cv::Point2f pointer) {
  using namespace cv;
  using namespace std;
  Mat imgResize, imgRotate, imgGray, imgBinary, srcPolar, polor180;
  float k, angle, angle_ridio, needle;
  //part:11111---------求指针角度及占比
  if (pointer.x > center.x & pointer.y < center.y) {
    k = abs(pointer.y - center.y) / abs(pointer.x - center.x);
    angle = atan(k) * 57.29577;
  }


  else if (pointer.x < center.x & pointer.y <= center.y) {
    k = abs(pointer.y - center.y) / abs(pointer.x - center.x);
    angle = atan(-k) * 57.29577 + 180;
  }

  else if (pointer.x < center.x & pointer.y >= center.y) {
    k = abs(pointer.y - center.y) / abs(pointer.x - center.x);
    angle = atan(k) * 57.29577 + 180;
  }

  else if (pointer.x > center.x & pointer.y > center.y) {
    k = abs(pointer.y - center.y) / abs(pointer.x - center.x);
    angle = atan(-k) * 57.29577 + 360;
  }

  angle_ridio = angle / 360.00;   //保留3位小数
  __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "angle ridio:%f", angle_ridio);
  // cout<<"angle_ridio="<<angle_ridio<<endl;

//part:22222-----resize，旋转，灰度化，滤波-------------
  int y = image.cols;
  int x = image.rows;

  int a = 500;   //resize,行向大小
  float factor = float(a) / x;   //缩放因子
  int b = int(y * factor);  //resize,纵向大小

  Point2f new_center;
  new_center.x = factor * center.x;
  new_center.y = factor * center.y;

  resize(image, imgResize, Size(a, b), 0, 0, INTER_LINEAR);
  circle(imgResize, new_center, 30, Scalar(0, 255, 0), -1, 8, 0);
  // imshow("imgResize", imgResize);

  // 1.旋转-90°
  Mat temp1, temp2;
  flip(imgResize, temp1, -1);
  transpose(temp1, temp2);
  flip(temp2, imgRotate, 1);

  cvtColor(imgRotate, imgGray, cv::COLOR_BGR2GRAY);  //灰度化
  medianBlur(imgGray, imgGray, 3);    //模糊化，滤波参数可调
  // imshow("imgGray", imgGray);

  // cout<<"x="<<x<<endl;
  // cout<<"factor="<<factor<<endl;
  // cout<<"b="<<b<<endl;

//part:33333-------自适应阈值，极坐标变换，变换后拉伸
  threshold(imgGray, imgBinary, 0, 255, THRESH_BINARY + THRESH_OTSU);
  // imshow("imgBinary", imgBinary);

  linearPolar(imgBinary, srcPolar, new_center, int(r * factor),
              WARP_FILL_OUTLIERS + INTER_LINEAR);
  // imshow("srcPolar", srcPolar);

  //旋转90°，第二次旋转
  Mat temp3, temp4;
  flip(srcPolar, temp3, -1);
  transpose(temp3, temp4);
  flip(temp4, polor180, 1);
  // imshow("polor180", polor180);

  //极坐标后的图像大小
  int polor_y = polor180.cols;
  int polor_x = polor180.rows;

  polor_x = polor_x + floor(polor_x / 2);

  Mat polor180r;
  resize(polor180, polor180r, Size(polor_x, polor_y), 0, 0, INTER_LINEAR);
  // imshow("polor180r", polor180r);


  // cout<<"polor_x="<<polor_x<<endl;

//part:444444-------求指针的纵坐标
  int polor_yr = polor180r.cols;
  int polor_xr = polor180r.rows;

  polor_xr = polor_xr - floor(polor_xr / 4);

  //上下剪切
  Mat polor180rr;
  Rect rect(0, 0, polor_yr, polor_xr);
  polor180r(rect).copyTo(polor180rr);


  // 拉伸90°后的极坐标图像
  // needle_quantity = np.sum(polor_180, axis=0)#纵向为0
  // needle = np.argmin(needle_quantity)#指针最小
  needle = polor_yr * angle_ridio;
  //cout << polor_yr << endl << angle_ridio << endl;
  // imshow("polor180rr", polor180rr);
  // cout<<"polor180rr.rows="<<polor180rr.rows<<endl;
  // cout<<"polor180rr.cols="<<polor180rr.cols<<endl;

  // cout<<"needle="<<needle<<endl;


//part:55555------------腐蚀（小卷积核） +   膨胀（大卷积核）-------
  Mat elementD = getStructuringElement(MORPH_RECT, Size(1, 1));
  dilate(polor180rr, polor180rr, elementD);
  // imshow("dilate",polor180rr);

  Mat elementE = getStructuringElement(MORPH_RECT, Size(13, 13));
  erode(polor180rr, polor180rr, elementE);
  // imshow("erode",polor180rr);


//part:66666----------滤除数字 : 找整体外部轮廓 + 生成一张纯白图（大小和polor_180相同）+相加法去除文字 + 腐蚀回去
  //整体外部轮廓
  vector <vector<Point>> contours;
  vector <Vec4i> hierarchy;
  findContours(polor180rr, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
  // imshow("cou", polor180rr);
////////////////截止到这里与python结果一模一样

  //生成一张白图
  int polor_yw = polor180rr.cols;
  int polor_xw = polor180rr.rows;
  Mat imgWhite(polor_xw, polor_yw, CV_8UC3, Scalar::all(255));

  //在新图上填充外部轮廓
  drawContours(imgWhite, contours, -1, (0, 0, 0), FILLED);
  //imshow("cou", imgWhite);

  //将新图灰度化
  cvtColor(imgWhite, imgWhite, cv::COLOR_BGR2GRAY);
  // imshow("imgWhite1", imgWhite);

  //将新图和二值膨胀图相加
  Mat imgAdd;
  imgAdd = imgWhite + polor180rr;
  // imshow("imgAdd", imgAdd);

  //再进行一次腐蚀
  Mat imgNotext;
  Mat elementD2 = getStructuringElement(MORPH_RECT, Size(13, 13));
  dilate(imgAdd, imgNotext, elementD2);
  // imshow("imgNotext",imgNotext);


//part:77777--》》》--求起始刻度
  //边缘检测
  Mat cannyPolar;
  Canny(imgNotext, cannyPolar, 100, 200, 5);
  //imshow("cannyPolar",cannyPolar);
  float score_final=0, score, left, right, score_range;

  //直线检测
  vector<Vec2f>lines;  //定义一个矢量结构lines用于存放得到的线段矢量集合
  HoughLines(cannyPolar, lines, 1, CV_PI / 180, 30);

  //依次在图中做出每条线段
  vector<int>obj;
  for (size_t i = 0; i < lines.size(); i++)
  {
    float r = lines[i][0];
    float theta = lines[i][1];

    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * r, y0 = b * r;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));

    //line(dstCanny, pt1, pt2, Scalar(255, 0, 0),1);

    //直线筛选，只保留横向直线
    if ((abs(pt1.x - pt2.x) >= 0) & (abs(pt1.x - pt2.x) <= 20)) {
      line(cannyPolar, pt1, pt2, Scalar(255, 0, 0), 1);
      //imshow("cannyPolar", cannyPolar);
      int x = pt1.x + (abs(pt2.x - pt1.x)) / 2;
      obj.push_back(x);
    }

  }

  sort(obj.begin(),obj.end());//从小到大
  //cout << obj.end() - obj.begin() << endl;
//    __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "obj size:%d", obj.size());
  if (obj.size() <2)
  {
    left = polor_yr * 0.125;
    right = polor_yr - left;
    score_range = (needle - left) / (right - left) * (end-start)+start;
    score_final = score_range;
    return score_final;
  }
//    __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "hello world");
  //刻度最大最小值
  float x_max = *max_element(obj.begin(), obj.end());
  float x_min = *min_element(obj.begin(), obj.end());


  //cout<<"x_max="<<x_max<<endl;
  //cout<<"x_min="<<x_min<<endl;



//part:88888---》》求读数


  score = (needle - x_min) / (x_max - x_min) * (1.025 * (end-start))+start;
  //cout << "score1=" << score << endl;

  left = polor_yr * 0.125;
  right = polor_yr - left;
  score_range = (needle - left) / (right - left) * (end-start)+start;
  //cout << "left" << left << "right" << right << "needle" << needle << "score" << score << "x_max" << x_max << "x_min" << x_min << endl;
  /*cout << "score2=" << score_range << endl;

  cout << "error ratio1:"<<(abs(score - score_range)) / abs(end-start)<< endl;
  cout << "error ratio2:"<<(abs(score - score_range)) / abs(score + score_range)<< endl;
  cout << "error ratio3:" << abs(score - score_range) / score_range << endl;*/

//    __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "score1 : %f score2:%f", score,score_range);
  if ((abs(score - score_range)) / abs(end - start) <= 0.03) {
    score_final = score;
  }     //误差系数
  else if (score < 0) {
    score_final = score_range;
  }

  else
    score_final = score_range;
  //若误差过大则返回score_range
  return score_final;
}

//寻找最大面积
std::vector<cv::Point> findBiggestContour(std::vector<std::vector<cv::Point>> contours)
{
  using namespace std;
  using namespace cv;

  vector<Vec4i> hierarchy;

  double largest_area = 0;
  int largest_contour_index = 0;

//  findContours(binary_image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
  __android_log_print(ANDROID_LOG_INFO, "Find_Center", "find contours ok");
  __android_log_print(ANDROID_LOG_INFO, "Find_Center", "contours size %d",contours.size());
  for (int i = 0; i < contours.size(); i++) // iterate through each contour.
  {
    double a = contourArea(contours[i], false);  //  Find the area of contour
    __android_log_print(ANDROID_LOG_INFO, "Find_Center", "contour %d area ok");
    if (a > largest_area) {
      largest_area = a;
      largest_contour_index = i;                //Store the index of largest contour
    }
  }

  return contours[largest_contour_index];
}
//寻找最大面积
std::vector<cv::Point> findBiggestContour2(cv::Mat& binary_image)
{
  using namespace std;
  using namespace cv;
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;

  double largest_area = 0;
  int largest_contour_index = 0;

  findContours(binary_image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  for (int i = 0; i < contours.size(); i++) // iterate through each contour.
  {
    double a = contourArea(contours[i], false);  //  Find the area of contour
    if (a > largest_area) {
      largest_area = a;
      largest_contour_index = i;                //Store the index of largest contour
    }
  }

  return contours[largest_contour_index];
}



//寻找最大面积
std::vector<cv::Point> findHighContour(cv::Mat& binary_image)
{
  using namespace std;
  using namespace cv;
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  Point2f center;
  float radius;
  double largest_area = 0;
  int largest_contour_index = 0;

  findContours(binary_image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  for (int i = 0; i < contours.size(); i++) // iterate through each contour.
  {
    double a = contourArea(contours[i], false);  //  Find the area of contour
    minEnclosingCircle(contours[i], center, radius);
    double value = a / (radius * radius * 3.14) * radius;
    if (value > largest_area) {
      largest_area = value;
      largest_contour_index = i;                //Store the index of largest contour
    }
  }

  return contours[largest_contour_index];
}
float distance2(cv::Point2f a, cv::Point2f b)
{
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

//寻找指针:距离中心最远的点
cv::Point find_point(std::vector<cv::Point> contour, cv::Point center)
{
  cv::Point res;
  float max_distance = 0;
  for (int i = 0; i < contour.size(); i++) // iterate through each contour.
  {
    float a = distance2(contour[i], center);  //  Find the area of contour
    if (a > max_distance) {
      max_distance = a;
      res = contour[i];                //Store the index of largest contour
    }
  }
  return res;
}
//// 寻找极坐标变换中心
//std::vector<float> find_center(cv::Mat imgBinary,cv::Point2f center,float radius) {
//  using namespace cv;
//  using namespace std;
//  vector<float> res;
//  Mat  imgDilate, zeroDilate;
//
//  //二值化
//  // adaptiveThreshold(imgGray, imgBinary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 25, 10);
////  threshold(imgGray, imgBinary, 0, 255, THRESH_BINARY + THRESH_OTSU);
//
//  //轮廓检测
////  vector <vector<Point>> contours;
////  vector <Vec4i> hierarchy;
////  findContours(imgBinary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
//
////  //膨胀
//  Mat elementD = getStructuringElement(MORPH_RECT, Size(3, 3));
//  dilate(imgBinary, imgDilate, elementD);
//
//  //找最大闭合区域的外接圆
////  vector<Point>  biggest_contour;
////  biggest_contour = findHighContour(imgBinary);
////
////
////  //寻找最小包围圆形
////  Point2f center;
////  float radius;
////  minEnclosingCircle(biggest_contour, center, radius);
//
//  //创建与imgDilate同等大小的零矩阵
//  zeroDilate = Mat::zeros(imgDilate.rows, imgDilate.cols, imgDilate.type());
//  circle(zeroDilate, center, int(radius * 0.8), (255, 255, 255), -1);
//  __android_log_print(ANDROID_LOG_INFO, "Find_Center", "dilate ok");
//  //矩阵点乘
//  Mat point_mat = (255-imgDilate).mul(zeroDilate);
//  __android_log_print(ANDROID_LOG_INFO, "Find_Center", "mul ok");
//
////  Mat point_mat2;
////  dilate(point_mat, point_mat2, elementD);
//  //imshow("point_mat",point_mat);
//  // waitKey(0);
//  //imshow("point", point_mat);
//  vector <vector<Point>> point_contours;
//  vector <Vec4i> point_hierarchy;
//  //findContours(point_mat, point_contours, point_hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
//  //找最大闭合区域的外接圆
//  vector<Point>  point_biggest_contour;
//  point_biggest_contour = findBiggestContour(point_mat);
//  __android_log_print(ANDROID_LOG_INFO, "Find_Center", "find big contour ok");
//  //vector<vector<Point>> b_contour;
//  //b_contour.push_back(point_biggest_contour);
//  //drawContours(imgGray, b_contour, -1, Scalar(255, 255, 255), 2);
//  //imshow("contour", imgGray);
//  // 计算指针的形心, 作为极坐标变换的中心
//
//  Moments M = moments(point_biggest_contour);
//  __android_log_print(ANDROID_LOG_INFO, "Find_Center", "find moments ok");
//  Point2f dstCenter;
//
//  dstCenter.x = int(M.m10 / M.m00);
//  dstCenter.y = int(M.m01 / M.m00);
//
//  Point pt=find_point(point_biggest_contour, dstCenter);
//  __android_log_print(ANDROID_LOG_INFO, "Find_Center", "find point ok");
//  //cout << pt << endl;
//
//  res.push_back(dstCenter.x);
//  res.push_back(dstCenter.y);
//  res.push_back(radius);
//  res.push_back(pt.x);
//  res.push_back(pt.y);
//
//  return res;
//
//}
std::vector<float> find_center(cv::Mat& imgGray) {
  using namespace  std;
  using namespace cv;
  vector<float> res;
  Mat imgBinary, imgDilate, zeroDilate;

  //二值化
  // adaptiveThreshold(imgGray, imgBinary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 25, 10);
  threshold(imgGray, imgBinary, 0, 255, THRESH_BINARY + THRESH_OTSU);
  __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "thresh");
  //轮廓检测
  vector <vector<Point>> contours;
  vector <Vec4i> hierarchy;
  findContours(imgBinary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
  __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "find");
  //膨胀
  Mat elementD = getStructuringElement(MORPH_RECT, Size(3, 3));
  dilate(imgBinary, imgDilate, elementD);
  __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "dilate");
  //找最大闭合区域的外接圆
  vector<Point>  biggest_contour;
  biggest_contour = findBiggestContour2(imgBinary);
  __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "find big");

  //寻找最小包围圆形
  Point2f center;
  float radius;
  minEnclosingCircle(biggest_contour, center, radius);
  __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "minEnclosingCircle");
  //创建与imgDilate同等大小的零矩阵
  zeroDilate = Mat::zeros(imgDilate.rows, imgDilate.cols, imgDilate.type());
  circle(zeroDilate, center, int(radius * 0.8), (255, 255, 255), -1);
  ;
  //矩阵点乘
  Mat point_mat = (255-imgDilate).mul(zeroDilate);
  __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "point_mat");
//  imshow("point_mat",point_mat);
  // waitKey(0);
  //imshow("point", point_mat);
  vector <vector<Point>> point_contours;
  vector <Vec4i> point_hierarchy;
  //findContours(point_mat, point_contours, point_hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
  //找最大闭合区域的外接圆
  vector<Point>  point_biggest_contour;
  point_biggest_contour = findBiggestContour2(point_mat);

  __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "point_mat");
  //vector<vector<Point>> b_contour;
  //b_contour.push_back(point_biggest_contour);
  //drawContours(imgGray, b_contour, -1, Scalar(255, 255, 255), 2);
  //imshow("contour", imgGray);
  // 计算指针的形心, 作为极坐标变换的中心
  Moments M = moments(point_biggest_contour);

  Point2f dstCenter;

  dstCenter.x = int(M.m10 / M.m00);
  dstCenter.y = int(M.m01 / M.m00);

  Point pt=find_point(point_biggest_contour, dstCenter);
  //cout << pt << endl;

  res.push_back(dstCenter.x);
  res.push_back(dstCenter.y);
  res.push_back(radius);
  res.push_back(pt.x);
  res.push_back(pt.y);

  return res;

}
//表盘阴影评估, 返回值返回0-1, 越大表明图像质量越好
float cal_possibility(cv::Mat& imgBinary,cv::Point2f &center,float &radius) {
  using namespace cv;
  using namespace std;
  Mat  imgDilate, zeroDilate;

  //二值化
  // adaptiveThreshold(imgGray, imgBinary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 25, 10);

  //imshow("gray", imgBinary);
  //轮廓检测
  vector <vector<Point>> contours;
  vector <Vec4i> hierarchy;
  findContours(imgBinary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

  //找最大闭合区域的外接圆
  vector<Point>  biggest_contour;
  biggest_contour = findHighContour(imgBinary);
  //drawContours(imgGray, biggest_contour, -1, Scalar(255, 255, 255), 1);
  //imshow("gray", imgGray);
  //寻找最小包围圆形
//  Point2f center;
//  float radius;
  minEnclosingCircle(biggest_contour, center, radius);

  return contourArea(biggest_contour) / (radius * radius * 3.14);
}






extern "C" JNIEXPORT jfloat JNICALL
Java_com_baidu_paddle_lite_demo_ocr_OCRPredictorNative_setbitmap(
        JNIEnv *env, jobject thiz, jobject original_image,jfloat angle,jfloat start,jfloat end) {
    using namespace cv;
    using namespace std;
//这块可以加你的处理代码么？然后给我返回一个float

//这里我需要多获取一个参数   Predictor 里 runModel 函数中 cal_angle的值
  cv::Mat origin = bitmap_to_cv_mat(env, original_image);

  Mat gray,imgBinary;
//  灰度化
  cvtColor(origin,gray,COLOR_BGR2GRAY);
// 二值化
  threshold(gray, imgBinary, 0, 255, THRESH_BINARY + THRESH_OTSU);
  Point2f center, pointp;
  float r;
// 阴影和圆心检测
  float poss=cal_possibility(imgBinary,center,r);
  __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "pos : %f start:%f end:%f", poss,start,end);
  if (poss<0.8) return 12345.5f;
  vector<float> points;
// 截取表盘位置的矩形区域
  int dx, dy;
  int w,h;
  dx = (int)center.x - (int)r;
  dy = (int)center.y - (int)r;
  if (dx+(int)(2*r)>=gray.cols){
    w=gray.cols-dx-2;
  }else w=2*(int)r;

  if(dy+(int)(2*r)>=gray.rows) h=gray.rows-dy-2;
  else h=2*(int)r;
  if (dx<0) dx=0;
  if (dy<0) dy=0;
  if (w<0 || h<0 || dx==0|| dy==0) return 10086.111f;
  Mat ins_mat;

  
  __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "size %d %d", gray.cols,gray.rows);
  __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "rect %d %d %d %d", dx,dy,w,h);
  Rect rect(dx, dy, w, h);
  gray(rect).copyTo(ins_mat);
  __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "cut image ok");


  if (poss>0.8 && angle<1){
//    找表盘圆心和指针顶点
    points= find_center(ins_mat);
    center.x = points[1];
    center.y = points[0];

    r = points[2];
    pointp.x = points[4];
    pointp.y = points[3];
    float dis=std::sqrt(distance2(center,pointp));
    __android_log_print(ANDROID_LOG_INFO, "Round_Meter", "center(%f,%f) point(%f,%f) dis :%f  0.8r:%f", center.x,center.y,pointp.x,pointp.y,dis,0.8*r);

    if (std::sqrt(distance2(center,pointp))>0.65*r)
        return round_meter(origin, start, end, center, (int)r, pointp);
    else {

      return 9999.5f;
    }
//    return 0;
  }else{
    if (poss<0.8)
      return 12345.5f;
    else if(angle>1)
      return 45678.5f;
  }

//  return cpp_array_to_jfloatarray(env, float_arr.data(), float_arr.size());

//  return 10086.111;
}


extern "C" JNIEXPORT void JNICALL
Java_com_baidu_paddle_lite_demo_ocr_OCRPredictorNative_release(
    JNIEnv *env, jobject thiz, jlong java_pointer) {
  if (java_pointer == 0) {
    LOGE("JAVA pointer is NULL");
    return;
  }
  ppredictor::OCR_PPredictor *ppredictor =
      (ppredictor::OCR_PPredictor *)java_pointer;
  delete ppredictor;
}