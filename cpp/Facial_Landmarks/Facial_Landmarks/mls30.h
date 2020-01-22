#include <opencv2/opencv.hpp>

#define GRID 30
#define	IMG_MAX_X	3000
#define	IMG_MAX_Y	2000
#define MAXPOINT	100

using namespace cv;
using namespace std;

static int calc_map = 0;

static Point2f	ptmap[IMG_MAX_Y/GRID+1][IMG_MAX_X/GRID+1];

static int calcMLS( vector<Point2f> &src, vector<Point2f> &dst );
static int calcMLS( vector<Point2f> &src, vector<Point2f> &dst, int xsize, int ysize );
static int MLSProjectionFast( int x, int y, float &tx, float &ty );
static int MLSProjectionSingle( vector<Point2f> &src, vector<Point2f> &dst, int x, int y, float &tx, float &ty );
// Warp image : mode = 0 (fast processing), 1 (fine processing but slow)
static void MLSWarpImage( Mat *src, vector<Point2f> &spts, Mat *dst, vector<Point2f> &dpts, int mode );

static int calcMLS( vector<Point2f> &src, vector<Point2f> &dst )
{
	return calcMLS( src, dst, IMG_MAX_X, IMG_MAX_Y );
}

static int calcMLS( vector<Point2f> &src, vector<Point2f> &dst, int xsize, int ysize )
{
	if( xsize > IMG_MAX_X  || ysize > IMG_MAX_Y ){
		printf("xsize or ysize is larger than maximum size (%d,%d)/(%d,%d)\n", xsize, ysize, IMG_MAX_X, IMG_MAX_Y );
		return 0;
	}

	float tx, ty;

	for( int y = 0; y < ysize/GRID + 2; y++ ){
		for( int x = 0; x < xsize/GRID + 2; x++ ){
			MLSProjectionSingle( src, dst, x*GRID, y*GRID, tx, ty );
			ptmap[y][x].x = tx;
			ptmap[y][x].y = ty;
		}
	}
	calc_map = 1;

	return 1;
}


static int MLSProjectionFast( int x, int y, float &tx, float &ty )
{
	if( calc_map == 0 ){
		printf("calcMLS() must be called before MLSProjectionFast()\n");
		return 0;
	}

	Point2f f00, f10, f01, f11;
	int   xx, yy;
	float dx, dy;

	f00 = ptmap[y/GRID][x/GRID];
	f01 = ptmap[y/GRID][x/GRID+1];
	f10 = ptmap[y/GRID+1][x/GRID];
	f11 = ptmap[y/GRID+1][x/GRID+1];

	dx = (float)(x - GRID*(x/GRID))/(float)GRID;
	dy = (float)(y - GRID*(y/GRID))/(float)GRID;

	tx = ( f00.x*(1.0-dy) + f10.x*dy )*(1.0-dx) + ( f01.x*(1.0-dy) + f11.x*dy )*dx;
	ty = ( f00.y*(1.0-dy) + f10.y*dy )*(1.0-dx) + ( f01.y*(1.0-dy) + f11.y*dy )*dx;

	return 1;
}


static int MLSProjectionSingle( vector<Point2f> &src, vector<Point2f> &dst, int x, int y, float &tx, float &ty )
{
	float W[MAXPOINT];			
	float wsum = 0.0;
	Point2f	pStar, qStar;		
	Point2f	pHat[MAXPOINT], qHat[MAXPOINT];		

	Mat A[MAXPOINT];				
	
	Mat P(2,2,CV_32F,0.0);			
	Mat V(2,2,CV_32F,0.0);
	Mat Vt(2,2,CV_32F,0.0);
	Mat Q(1,2,CV_32F,0.0);

	for( int i = 0; i < src.size(); i++ ){
		W[i] = 1.0 / ( pow( float(x - src[i].x) + 0.5, 2) + pow( float(y-src[i].y) +  0.5, 2) );
		wsum += W[i];
	}

	pStar.x = 0.0; pStar.y = 0.0;
	qStar.x = 0.0; qStar.y = 0.0;
	for( int j = 0; j < src.size(); j++ ){
		pStar.x += ( W[j] * src[j].x );
		pStar.y += ( W[j] * src[j].y );
		qStar.x += ( W[j] * dst[j].x );
		qStar.y += ( W[j] * dst[j].y );
	}
	qStar /= wsum;
	pStar /= wsum;

	for( int i = 0; i < src.size(); i++ ){
		pHat[i].x = src[i].x - pStar.x;
		pHat[i].y = src[i].y - pStar.y;
		qHat[i].x = dst[i].x - qStar.x;
		qHat[i].y = dst[i].y - qStar.y;
	}

	for( int i = 0; i < src.size(); i++ )
	{
        P.at<float>(0, 0) = pHat[i].x;
        P.at<float>(0, 1) = pHat[i].y;
        P.at<float>(1, 0) = pHat[i].y;
        P.at<float>(1, 1) = -pHat[i].x;

        V.at<float>(0, 0) = x - pStar.x;
        V.at<float>(0, 1) = y - pStar.y;
        V.at<float>(1, 0) = y - pStar.y;
        V.at<float>(1, 1) = -(x - pStar.x);

		transpose(V, Vt);

		A[i] = W[i] * P * Vt;
	}

	Mat Fr(1,2,CV_32F,0.0);
	Mat tempFr(1,2,CV_32F,0.0);
	float lenFr, dist;

	for( int i = 0; i < src.size(); i++ )
	{
		Q.at<float>(0,0) = qHat[i].x;
		Q.at<float>(0,1) = qHat[i].y;

		tempFr = Q * A[i];
		
		Fr += tempFr;
	}

	lenFr = sqrt( powf(Fr.at<float>(0, 0),2) + powf(Fr.at<float>(0, 1),2));

	Fr /= lenFr;
	
	
	dist = sqrt( (x-pStar.x)*(x-pStar.x) + (y-pStar.y)*(y-pStar.y) );

	tx = dist * Fr.at<float>(0, 0) + qStar.x;
	ty = dist * Fr.at<float>(0, 1) + qStar.y;

	return 1;
}


static void MLSWarpImage( Mat &src, vector<Point2f> &spts, Mat &dst, vector<Point2f> &dpts, int mode )
{
	float tx, ty;
	int xcoord, ycoord;

	if( mode == 0 ){
		calcMLS( dpts, spts, dst.cols, dst.rows );
	}

	for( int y = 0; y < dst.rows; y++ )
	{
		for( int x = 0; x < dst.cols; x++ )
		{
			if( mode == 0 )
				MLSProjectionFast( x, y, tx, ty );
			else 
				MLSProjectionSingle( dpts, spts, x, y, tx, ty );

			if( tx < 0 || tx > src.cols-1 || ty < 0 || ty > src.rows-1 )
			{
				
				dst.at<Vec3b>(y,x) = Vec3b(0,0,0);
			} 
			else 
			{
				xcoord = (int)(tx+0.5);
				ycoord = (int)(ty+0.5);
				if (xcoord > src.cols-1 || xcoord < 0 ||
				    ycoord > src.rows-1 || ycoord < 0)
				{
				  dst.at<Vec3b>(y,x) = Vec3b(0,0,0);
				}
				else
				{
				  dst.at<Vec3b>(y,x) = src.at<Vec3b>(ycoord, xcoord);
				}
			}
		}
	}
} 
