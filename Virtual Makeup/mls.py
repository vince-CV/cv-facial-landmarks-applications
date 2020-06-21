import numpy as np

GRID = 80
IMG_MAX_X = 3000
IMG_MAX_Y = 2000
MAXPOINT = 100
ptmap = np.zeros(((int(IMG_MAX_Y/GRID)+1),(int(IMG_MAX_X/GRID)+1),2), dtype=np.float32)

def calcMLS( src, dst, xsize, ysize ):
  #Create Map for Projection
  if( xsize > IMG_MAX_X  or ysize > IMG_MAX_Y ):
    print("xsize or ysize is larger than maximum size ({},{})/({},{})".format(xsize, ysize, IMG_MAX_X, IMG_MAX_Y))
    return 0
  
  # Project GRID Points
  for y in range(0,int(ysize/GRID) + 2) :
    for x in range( 0,int(xsize/GRID) + 2) :
      tx,ty = MLSProjectionSingle( src, dst, x*GRID, y*GRID )
      ptmap[y,x,0] = tx
      ptmap[y,x,1] = ty
      
  return 1


def MLSProjectionFast( x, y ):

  #unit square
  f00 = ptmap[np.uint32(y/GRID),np.uint32(x/GRID)]
  f01 = ptmap[np.uint32(y/GRID),np.uint32(x/GRID)+1]
  f10 = ptmap[np.uint32(y/GRID)+1,np.uint32(x/GRID)]
  f11 = ptmap[np.uint32(y/GRID)+1,np.uint32(x/GRID)+1]

  dx = (x - GRID*(np.uint32(x/GRID)))/np.float32(GRID)
  dy = (y - GRID*(np.uint32(y/GRID)))/np.float32(GRID)

  # bilinear interpolation
  tx = ( f00[:,0]*(1.0-dy) + f10[:,0]*dy )*(1.0-dx) + ( f01[:,0]*(1.0-dy) + f11[:,0]*dy )*dx
  ty = ( f00[:,1]*(1.0-dy) + f10[:,1]*dy )*(1.0-dx) + ( f01[:,1]*(1.0-dy) + f11[:,1]*dy )*dx

  return tx,ty


def MLSProjectionSingle( src, dst, x, y ):

  # Intermediate matrices for computation
  P = np.zeros((2,2),dtype=np.float32)
  V = np.zeros((2,2),dtype=np.float32)
  Vt = np.zeros((2,2),dtype=np.float32)
  Q = np.zeros(2,dtype=np.float32)
  
  # calc weights
  W = 1.0 / ( np.power( np.float32(x - src[:,0]) + 0.5, 2) + np.power( np.float32(y-src[:,1]) +  0.5, 2) )
  wsum = np.sum(W)

  # calculate centroids of p,q w.r.t W --> p* and q*
  pStar = np.dot( W , src )
  qStar = np.dot( W , dst )
  
  qStar /= wsum
  pStar /= wsum

  # calc phat and qhat -- p^ and q^
  pHat = src - pStar
  qHat = dst - qStar
  
  # calc Ai
  A = np.zeros((2,2,MAXPOINT),dtype=np.float32)
  for i in range(0 , src.shape[0]) :
  
    P[0, 0] = pHat[i][0]
    P[0, 1] = pHat[i][1]
    P[1, 0] = pHat[i][1]
    P[1, 1] = -pHat[i][0]

    V[0, 0] = x - pStar[0]
    V[0, 1] = y - pStar[1]
    V[1, 0] = y - pStar[1]
    V[1, 1] = -(x - pStar[0])

    Vt = np.transpose(V)

    A[:,:,i] = W[i] * np.dot( P , Vt);

  Fr = np.zeros( 2, dtype=np.float32 )

  # Calc Fr and |Fr|
  for i in range(0, src.shape[0]) :
  
    Q[0] = qHat[i][0]
    Q[1] = qHat[i][1]

    tempFr = np.dot(Q , A[:,:,i])

    Fr += tempFr

  lenFr = np.sqrt( np.power(Fr[0],2) + np.power(Fr[1],2))

  Fr /= lenFr;

  # Calc |V - p*|
  dist = np.sqrt( (x-pStar[0])*(x-pStar[0]) + (y-pStar[1])*(y-pStar[1]) )

  tx = dist * Fr[0] + qStar[0]
  ty = dist * Fr[1] + qStar[1]

  return tx,ty


def MLSWarpImage( src, spts, dpts):
  dst = np.zeros(src.shape,src.dtype)

  # Precompute Map for dpts --> spts
  calcMLS( dpts, spts, dst.shape[1], dst.shape[0] )

  yy,xx = np.mgrid[0:dst.shape[0],0:dst.shape[1]]
  xx = xx.reshape(xx.shape[0] * xx.shape[1])
  yy = yy.reshape(yy.shape[0] * yy.shape[1])
  txx,tyy = MLSProjectionFast( xx, yy )
  
  xx = xx.reshape(dst.shape[0],dst.shape[1])
  yy = yy.reshape(dst.shape[0],dst.shape[1])

  txx = txx.reshape(dst.shape[0],dst.shape[1])
  tyy = tyy.reshape(dst.shape[0],dst.shape[1])
  
  txx = np.int32(txx + 0.5)
  tyy = np.int32(tyy + 0.5)
  
  src[0,0] = [0,0,0]
  
  txx = txx*(txx>0)
  tyy = tyy*(tyy>0)
  txx = txx*(txx<src.shape[1])
  tyy = tyy*(tyy<src.shape[0])

  dst[yy,xx]=src[tyy,txx]
  
  return dst
