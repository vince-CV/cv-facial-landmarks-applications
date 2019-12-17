
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

using namespace cv;
using namespace std;

#ifndef M_PI
#define M_PI 3.14159
#endif



static void constrainPoint(Point2f &p, Size sz)
{
	p.x = min(max((double)p.x, 0.0), (double)(sz.width - 1)); 
	p.y = min(max((double)p.y, 0.0), (double)(sz.height - 1));

}


static void getEightBoundaryPoints(Size size, vector<Point2f>& boundaryPts)
{
	int h = size.height, w = size.width;
	boundaryPts.push_back(Point2f(0, 0));
	boundaryPts.push_back(Point2f(w / 2, 0));
	boundaryPts.push_back(Point2f(w - 1, 0));
	boundaryPts.push_back(Point2f(w - 1, h / 2));
	boundaryPts.push_back(Point2f(w - 1, h - 1));
	boundaryPts.push_back(Point2f(w / 2, h - 1));
	boundaryPts.push_back(Point2f(0, h - 1));
	boundaryPts.push_back(Point2f(0, h / 2));
}


static void dlibLandmarksToPoints(dlib::full_object_detection &landmarks, vector<Point2f>& points)
{
	for (int i = 0; i < landmarks.num_parts(); i++)
	{
		Point2f pt(landmarks.part(i).x(), landmarks.part(i).y());
		points.push_back(pt);
	}
}

static void similarityTransform(std::vector<cv::Point2f>& inPoints, std::vector<cv::Point2f>& outPoints, cv::Mat &tform)
{
	double s60 = sin(60 * M_PI / 180.0);
	double c60 = cos(60 * M_PI / 180.0);

	vector <Point2f> inPts = inPoints;
	vector <Point2f> outPts = outPoints;

	inPts.push_back(cv::Point2f(0, 0));
	outPts.push_back(cv::Point2f(0, 0));

	inPts[2].x = c60 * (inPts[0].x - inPts[1].x) - s60 * (inPts[0].y - inPts[1].y) + inPts[1].x;
	inPts[2].y = s60 * (inPts[0].x - inPts[1].x) + c60 * (inPts[0].y - inPts[1].y) + inPts[1].y;

	outPts[2].x = c60 * (outPts[0].x - outPts[1].x) - s60 * (outPts[0].y - outPts[1].y) + outPts[1].x;
	outPts[2].y = s60 * (outPts[0].x - outPts[1].x) + c60 * (outPts[0].y - outPts[1].y) + outPts[1].y;

	tform = cv::estimateAffinePartial2D(inPts, outPts);
}

static void normalizeImagesAndLandmarks(Size outSize, Mat &imgIn, Mat &imgOut, vector<Point2f>& pointsIn, vector<Point2f>& pointsOut)
{
	int h = outSize.height;
	int w = outSize.width;


	vector<Point2f> eyecornerSrc;
	if (pointsIn.size() == 68)
	{
		eyecornerSrc.push_back(pointsIn[36]);
		eyecornerSrc.push_back(pointsIn[45]);
	}
	else if (pointsIn.size() == 5)
	{
		eyecornerSrc.push_back(pointsIn[2]);
		eyecornerSrc.push_back(pointsIn[0]);
	}

	vector<Point2f> eyecornerDst;
	eyecornerDst.push_back(Point2f(0.3*w, h / 3));
	eyecornerDst.push_back(Point2f(0.7*w, h / 3));

	Mat tform;
	similarityTransform(eyecornerSrc, eyecornerDst, tform);

	imgOut = Mat::zeros(h, w, CV_32FC3);
	warpAffine(imgIn, imgOut, tform, imgOut.size());

	transform(pointsIn, pointsOut, tform);

}

static int findIndex(vector<Point2f>& points, Point2f &point)
{
	int minIndex = 0;
	double minDistance = norm(points[0] - point);
	for (int i = 1; i < points.size(); i++)
	{
		double distance = norm(points[i] - point);
		if (distance < minDistance)
		{
			minIndex = i;
			minDistance = distance;
		}

	}
	return minIndex;
}



static void calculateDelaunayTriangles(Rect rect, vector<Point2f> &points, vector< vector<int> > &delaunayTri) {

	Subdiv2D subdiv(rect);

	for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
		subdiv.insert(*it);

	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);

	vector<Point2f> pt(3);
	vector<int> ind(3);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];

		pt[0] = Point2f(t[0], t[1]);
		pt[1] = Point2f(t[2], t[3]);
		pt[2] = Point2f(t[4], t[5]);


		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			for (int j = 0; j < 3; j++)
			{
				ind[j] = findIndex(points, pt[j]);
			}
			delaunayTri.push_back(ind);
		}
	}

}

static void applyAffineTransform(Mat &warpImage, Mat &src, vector<Point2f> &srcTri, vector<Point2f> &dstTri)
{
	Mat warpMat = getAffineTransform(srcTri, dstTri);
	warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

static void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> t1, vector<Point2f> t2)
{

	Rect r1 = boundingRect(t1);
	Rect r2 = boundingRect(t2);

	vector<Point2f> t1Rect, t2Rect;
	vector<Point> t2RectInt;
	for (int i = 0; i < 3; i++)
	{
		t2RectInt.push_back(Point((int)(t2[i].x - r2.x), (int)(t2[i].y - r2.y))); // for fillConvexPoly

		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
	}


	Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
	fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

	Mat img1Rect, img2Rect;
	img1(r1).copyTo(img1Rect);


	Mat warpImage = Mat::zeros(r2.height, r2.width, img1Rect.type());

	applyAffineTransform(warpImage, img1Rect, t1Rect, t2Rect);

	// Copy triangular region of the rectangular patch to the output image
	multiply(warpImage, mask, warpImage);
	multiply(img2(r2), Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
	img2(r2) = img2(r2) + warpImage;

}

static bool rectAreaComparator(dlib::rectangle &r1, dlib::rectangle &r2)
{
	return r1.area() < r2.area();
}


static vector<Point2f> getLandmarks(dlib::frontal_face_detector &faceDetector, dlib::shape_predictor &landmarkDetector, Mat &img, float FACE_DOWNSAMPLE_RATIO = 1)
{

	vector<Point2f> points;

	Mat imgSmall;
	cv::resize(img, imgSmall, cv::Size(), 1.0 / FACE_DOWNSAMPLE_RATIO, 1.0 / FACE_DOWNSAMPLE_RATIO);

	// Convert OpenCV image format to Dlib's image format
	dlib::cv_image<dlib::bgr_pixel> dlibIm(img);
	dlib::cv_image<dlib::bgr_pixel> dlibImSmall(imgSmall);


	// Detect faces in the image
	std::vector<dlib::rectangle> faceRects = faceDetector(dlibImSmall);

	if (faceRects.size() > 0)
	{
		// Pick the biggest face
		dlib::rectangle rect = *std::max_element(faceRects.begin(), faceRects.end(), rectAreaComparator);

		dlib::rectangle scaledRect(
			(long)(rect.left() * FACE_DOWNSAMPLE_RATIO),
			(long)(rect.top() * FACE_DOWNSAMPLE_RATIO),
			(long)(rect.right() * FACE_DOWNSAMPLE_RATIO),
			(long)(rect.bottom() * FACE_DOWNSAMPLE_RATIO)
		);

		dlib::full_object_detection landmarks = landmarkDetector(dlibIm, scaledRect);
		dlibLandmarksToPoints(landmarks, points);
	}

	return points;

}



static void warpImage(Mat &imgIn, Mat &imgOut, vector<Point2f> &pointsIn, vector<Point2f> &pointsOut, vector< vector<int> > &delaunayTri)
{

	Size size = imgIn.size();
	imgOut = Mat::zeros(size, imgIn.type());

	for (size_t j = 0; j < delaunayTri.size(); j++)
	{

		vector<Point2f> tin, tout;

		for (int k = 0; k < 3; k++)
		{
			Point2f pIn = pointsIn[delaunayTri[j][k]];
			constrainPoint(pIn, size);


			Point2f pOut = pointsOut[delaunayTri[j][k]];
			constrainPoint(pOut, size);

			tin.push_back(pIn);

			tout.push_back(pOut);
		}


		warpTriangle(imgIn, imgOut, tin, tout);
	}
}




