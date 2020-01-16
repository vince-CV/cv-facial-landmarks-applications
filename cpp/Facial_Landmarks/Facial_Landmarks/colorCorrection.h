// Copyright (c) 2015 Matthew Earl
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
//     The above copyright notice and this permission notice shall be included
//     in all copies or substantial portions of the Software.
// 
//     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
//     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//     USE OR OTHER DEALINGS IN THE SOFTWARE.

Mat correctColours(Mat im1, Mat im2, std::vector<Point2f> points2)// lower number --> output is closer to webcam and vice-versa
{    
    

    Point2f dist_between_eyes =  points2[38] - points2[43]; 
    float distance = norm(dist_between_eyes);

    //using heuristics to calculate the amount of blur
    int blur_amount = int(0.5 * distance);

    if (blur_amount % 2 == 0)
        blur_amount += 1;

    Mat im1_blur = im1.clone();
    Mat im2_blur = im2.clone();

    cv::blur(im1_blur,im1_blur, Size (blur_amount, blur_amount));
    cv::blur(im2_blur,im2_blur, Size (blur_amount, blur_amount));
    // Avoid divide-by-zero errors.

    im2_blur += 2*(im2_blur <= 1)/255;

    im1_blur.convertTo(im1_blur, CV_32F);
    im2_blur.convertTo(im2_blur, CV_32F);
    im2.convertTo(im2, CV_32F);

    Mat ret = im2.clone();
    ret = im2.mul(im1_blur).mul(1/im2_blur);

    threshold(ret,ret,255,255,THRESH_TRUNC);

    ret.convertTo(ret,CV_8UC3);

    return ret;

}