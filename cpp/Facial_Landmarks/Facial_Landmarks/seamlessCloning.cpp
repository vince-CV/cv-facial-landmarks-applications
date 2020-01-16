#include <opencv2/opencv.hpp>
#include <iostream>
//#include "dataPath.hpp"

using namespace std;
using namespace cv;

int seamlessClone(){


	string DATA_PATH = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/";

	Mat src = imread(DATA_PATH + "images/airplane.jpg");
	Mat dst = imread(DATA_PATH + "images/sky.jpg");

	Mat src_mask = Mat::zeros(src.rows, src.cols, src.depth());
	Point poly[1][7];
	poly[0][0] = Point(4, 80);
	poly[0][1] = Point(30, 54);
	poly[0][2] = Point(151,63);
	poly[0][3] = Point(254,37);
	poly[0][4] = Point(298,90);
	poly[0][5] = Point(272,134);
	poly[0][6] = Point(43,122);

	const Point* polygons[1] = { poly[0] };
	int num_points[] = { 7 };

	fillPoly(src_mask, polygons, num_points, 1, Scalar(255,255,255));

	Point center(800,100);

	Mat output;
	seamlessClone(src, dst, src_mask, center, output, NORMAL_CLONE);
	imshow("Output",output);
	waitKey(0);


	Mat img = imread(DATA_PATH + "images/wood-texture.jpg");
	Mat obj = imread(DATA_PATH + "images/iloveyouticket.jpg");

	src_mask = 255 * Mat::ones(obj.rows, obj.cols, obj.depth());
	center = Point(img.cols / 2, img.rows / 2);

	Mat normal_clone;
	Mat mixed_clone;

	seamlessClone(obj, img, src_mask, center, normal_clone, NORMAL_CLONE);
	seamlessClone(obj, img, src_mask, center, mixed_clone, MIXED_CLONE);
	imshow("Normal Clone Result", normal_clone);
	imshow("Mixed Clone Result", mixed_clone);
	waitKey(0);

	destroyAllWindows();
	return 0;



}
