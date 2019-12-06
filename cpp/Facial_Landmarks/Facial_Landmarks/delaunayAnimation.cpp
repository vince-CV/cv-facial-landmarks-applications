#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;


static void drawPoint( Mat& img, Point2f fp, Scalar color )
{
  circle( img, fp, 2, color, FILLED, LINE_AA, 0 );
}


static void drawDelaunay( Mat& img, Subdiv2D& subdiv, Scalar delaunayColor )
{

  vector<Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);


  vector<Point> vertices(3);

  Size size = img.size();
  Rect rect(0,0, size.width, size.height);

  for( size_t i = 0; i < triangleList.size(); i++ )
  {

    Vec6f t = triangleList[i];


    vertices[0] = Point(cvRound(t[0]), cvRound(t[1]));
    vertices[1] = Point(cvRound(t[2]), cvRound(t[3]));
    vertices[2] = Point(cvRound(t[4]), cvRound(t[5]));


    if ( rect.contains(vertices[0]) && rect.contains(vertices[1]) && rect.contains(vertices[2]))
    {
      line(img, vertices[0], vertices[1], delaunayColor, 1, LINE_AA, 0);
      line(img, vertices[1], vertices[2], delaunayColor, 1, LINE_AA, 0);
      line(img, vertices[2], vertices[0], delaunayColor, 1, LINE_AA, 0);
    }
  }
}

static void drawVoronoi( Mat& img, Subdiv2D& subdiv )
{

  vector<vector<Point2f> > facets;

  vector<Point2f> centers;

  subdiv.getVoronoiFacetList(vector<int>(), facets, centers);


  vector<Point> ifacet;

  vector<vector<Point> > ifacets(1);

  for( size_t i = 0; i < facets.size(); i++ )
  {

    ifacet.resize(facets[i].size());
    for( size_t j = 0; j < facets[i].size(); j++ )
      ifacet[j] = facets[i][j];

    Scalar color;
    color[0] = rand() & 255;
    color[1] = rand() & 255;
    color[2] = rand() & 255;

    fillConvexPoly(img, ifacet, color, 8, 0);

    ifacets[0] = ifacet;
    polylines(img, ifacets, true, Scalar(), 1, LINE_AA, 0);

    circle(img, centers[i], 3, Scalar(), FILLED, LINE_AA, 0);
  }
}


static int findIndex(vector<Point2f>& points, Point2f &point)
{
  int minIndex = 0;
  double minDistance = norm(points[0] - point);
  for(int i = 1; i < points.size(); i++)
  {
    double distance = norm(points[i] - point);
    if( distance < minDistance )
    {
      minIndex = i;
      minDistance = distance;
    }

  }
  return minIndex;
}


static void writeDelaunay(Subdiv2D& subdiv, vector<Point2f>& points, const string &filename)
{


  std::ofstream ofs;
  ofs.open(filename);


  vector<Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);


  vector<Point2f> vertices(3);


  for( size_t i = 0; i < triangleList.size(); i++ )
  {

    Vec6f t = triangleList[i];

    vertices[0] = Point2f(t[0], t[1]);
    vertices[1] = Point2f(t[2], t[3]);
    vertices[2] = Point2f(t[4], t[5]);


    ofs << findIndex(points, vertices[0]) << " "
    << findIndex(points, vertices[1]) << " "
    << findIndex(points, vertices[2]) << endl;

  }
  ofs.close();
}


int main( int argc, char** argv)
{


  string win = "Delaunay Triangulation & Voronoi Diagram";


  Scalar delaunayColor(255,255,255), pointsColor(0, 0, 255);


  Mat img = imread("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/smiling-man.jpg");


  Size size = img.size();
  Rect rect(0, 0, size.width, size.height);


  Subdiv2D subdiv(rect);


  vector<Point2f> points;


  ifstream ifs("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/smiling-man-delaunay.txt");
  int x, y;
  while(ifs >> x >> y)
  {
    points.push_back(Point2f(x,y));
  }


  Mat imgDelaunay;


  Mat imgVoronoi = Mat::zeros(img.rows, img.cols, CV_8UC3);


  Mat imgDisplay;


  for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
  {
    subdiv.insert(*it);

    imgDelaunay = img.clone();
    imgVoronoi = cv::Scalar(0,0,0);

    drawDelaunay( imgDelaunay, subdiv, delaunayColor );


    for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
      drawPoint(imgDelaunay, *it, pointsColor);
    }


    drawVoronoi(imgVoronoi, subdiv);

    hconcat(imgDelaunay, imgVoronoi, imgDisplay);
    imshow(win, imgDisplay);
    waitKey(100);
  }


  writeDelaunay(subdiv, points, "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/results/smiling-man-delaunay.tri");


  waitKey(0);

  return EXIT_SUCCESS;
}
