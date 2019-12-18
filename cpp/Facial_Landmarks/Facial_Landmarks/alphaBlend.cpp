#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


Mat& blend(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage)
{
    Mat fore, back;
    multiply(alpha, foreground, fore);
    multiply(Scalar::all(1.0)-alpha, background, back);
    add(fore, back, outImage);

    return outImage;
}


Mat& alphaBlendDirectAccess(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage)
{

    int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();

    float* fptr = reinterpret_cast<float*>(foreground.data);
    float* bptr = reinterpret_cast<float*>(background.data);
    float* aptr = reinterpret_cast<float*>(alpha.data);
    float* outImagePtr = reinterpret_cast<float*>(outImage.data);

    int i,j;
    for ( j = 0; j < numberOfPixels; ++j, outImagePtr++, fptr++, aptr++, bptr++)
    {
        *outImagePtr = (*fptr)*(*aptr) + (*bptr)*(1 - *aptr);
    }

    return outImage;
}


int alphaBlend(int argc, char** argv)
{

	Mat foreGroundImage = imread("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/foreGroundAssetLarge.png", -1);
	Mat background = imread("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/backGroundLarge.jpg");

	Mat bgra[4];
    split(foreGroundImage, bgra); // BGR + alpha mask

    vector<Mat> foregroundChannels;
    foregroundChannels.push_back(bgra[0]);
    foregroundChannels.push_back(bgra[1]);
    foregroundChannels.push_back(bgra[2]);
    Mat foreground = Mat::zeros(foreGroundImage.size(), CV_8UC3);
    merge(foregroundChannels, foreground);

    vector<Mat> alphaChannels;
    alphaChannels.push_back(bgra[3]);
    alphaChannels.push_back(bgra[3]);
    alphaChannels.push_back(bgra[3]);
    Mat alpha = Mat::zeros(foreGroundImage.size(), CV_8UC3);
    merge(alphaChannels, alpha);

    Mat copyWithMask = Mat::zeros(foreGroundImage.size(), CV_8UC3);
    foreground.copyTo(copyWithMask, bgra[3]);
    
 
    foreground.convertTo(foreground, CV_32FC3);
    background.convertTo(background, CV_32FC3);
    alpha.convertTo(alpha, CV_32FC3, 1.0/255); // keeps value between 0 and 1

    int numOfIterations = 1; 


    Mat outImage= Mat::zeros(foreground.size(), foreground.type());
    double t = (double)getTickCount();
    for (int i=0; i<numOfIterations; i++) {
        outImage = blend(alpha, foreground, background, outImage);
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Time using multiply & add function : " << t*1000/numOfIterations << " milliseconds" << endl;

    // direct Mat access / pixel-wise 
    outImage = Mat::zeros(foreground.size(), foreground.type());
    t = (double)getTickCount();
    for (int i=0; i<numOfIterations; i++) {
        outImage = alphaBlendDirectAccess(alpha, foreground, background, outImage);
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Time using alphaBlendDirectAccess : " << t*1000/numOfIterations << " milliseconds" << endl;

    imshow("Alpha blended image", outImage/255);
    waitKey(0);

    return 0;
}
