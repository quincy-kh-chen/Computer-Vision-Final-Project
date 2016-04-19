/*
 16-720 project
 Purpose: MOSSE tracker
 
 @author Quincy Chen
 @email kunhsinc@andrew.cmu.edu
 @version 1 4/10/16
*/
#ifndef LATTICE_IS_INCLUDED
#define LATTICE_IS_INCLUDED

#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <opencv/cv.h>


using namespace cv;
using namespace std;
class Mosse 
{
public:
  double eps=0.0001;
  double rate=0.125; //learning rate
  Point_<double> center; //center of the bounding box
  Size size; //size of the bounding box
  Mat previousWin;
  Mat previousRes;
  Mat HanWin;

  //Optimal size for DFT processing  
  int h;
  int w;

  //Desired output
  Mat G;

  //For H
  Mat A;
  Mat B;
  Mat H;

  Mosse();
  ~Mosse();
  void CleanUp(void); 
  //visualizatiom
  void Draw(Mat &frame);
  
  //computing
  void Initialize(Mat &frame, Rect rect);
  void DefineGoal(void); 
  void PreProcess(Mat &window); 
  double Correlate(Mat iamge,Point &delta_xy);
  void UpdateFilter(void);
  Mat randWarp(const Mat &a);
  float randNum(void);
  Mat divDFTs(CvMat src1, CvMat src2);
  void Run(const Mat &frame);
  

};
Mosse::Mosse(void)
{
}
Mosse::~Mosse(void)
{
}
void Mosse::Draw(Mat &frame)
{
  double x=center.x;
  double y=center.y;
  double x1=int(x-0.5*w); 
  double y1=int(y-0.5*h); 
  double x2=int(x+0.5*w); 
  double y2=int(y+0.5*h);
  rectangle(frame,Point(x1,y1),Point(x2,y2), CV_RGB(0, 255, 0), 3, 8, 0);
  circle(frame, center, 2,CV_RGB(0, 255, 0),-1);
  imshow("Tracking",frame);
}
void Mosse::Initialize(Mat &frame, Rect rect)
{
  // CleanUp();
  //Get the optimal size for DFT processing
  w=getOptimalDFTSize(rect.width);
  h=getOptimalDFTSize(rect.height);
  
  //Get the center position
  int x1=floor((2*rect.x+rect.width-w)/2);
  int y1=floor((2*rect.y+rect.height-h)/2);
  center.x=x1+(w-1)/2;//
  center.y=y1+(h-1)/2;//
  size.width=w;
  size.height=h;
  
  //Initialize FFT
  Mat window;
  getRectSubPix(frame, size, center, window); //window is the output array
  
  //create Hanning window
  createHanningWindow(HanWin, size, CV_32F);

  //define G
  DefineGoal();

  //compute A,B and H
  //affine image
  A=B=Mat::zeros(G.size(), G.type()); // A.size()=B.size()=G.size()
  for(int i=0;i<8;i++)
  {
    Mat window_warp=randWarp(window);
    PreProcess(window_warp);

    Mat WINDOW_WARP;
    Mat A_i;
    Mat B_i;
    dft(window_warp,WINDOW_WARP,DFT_COMPLEX_OUTPUT);
    mulSpectrums(G          , WINDOW_WARP, A_i, 0, true );
    mulSpectrums(WINDOW_WARP, WINDOW_WARP, B_i, 0, true );
//    A+=A_i;
//    B+=B_i;
    add(A,A_i,A);
    add(B,B_i,B);
  }

  //update filter
  UpdateFilter();
  std::cout<<"Initialization completed"<<std::endl;
}

void Mosse::UpdateFilter(void)
{
  H=divDFTs(A,B);
  // H*=-1;
  // cout<<"********"<<H<<"********"<<endl;
}

Mat Mosse::divDFTs(CvMat src1, CvMat src2)
{ 
  /* Komponentenweise division der Werte in src1 und src2 
  */ 
  CvMat* x1_imageRe = cvCreateMat(size.height, size.width, CV_32FC1); 
  CvMat* x1_imageIm = cvCreateMat(size.height, size.width, CV_32FC1); 
  CvMat* x2_imageRe = cvCreateMat(size.height, size.width, CV_32FC1); 
  CvMat* x2_imageIm = cvCreateMat(size.height, size.width, CV_32FC1); 
  //CvMat* imageImRot = cvCreateMat(this->source_depth_dft->height,this->source_depth_dft->width, CV_32FC1); 
  cvSplit(&src1, x1_imageRe, x1_imageIm, 0, 0 ); 
  cvSplit(&src2, x2_imageRe, x2_imageIm, 0, 0 );

  // Divide real and imaginary components 
  CvMat* denominator1 = cvCreateMat(size.height, size.width, CV_32FC1); //nenner 
  CvMat* denominator2 = cvCreateMat(size.height, size.width, CV_32FC1); 
  CvMat* numerator1 = cvCreateMat(size.height, size.width, CV_32FC1); //zähler
  CvMat* numerator2 = cvCreateMat(size.height, size.width, CV_32FC1); 
  CvMat* addend1 = cvCreateMat(size.height, size.width, CV_32FC1); 
  CvMat* addend2 = cvCreateMat(size.height, size.width, CV_32FC1); 
  CvMat* square1 = cvCreateMat(size.height, size.width, CV_32FC1); 
  CvMat* square2 = cvCreateMat(size.height, size.width, CV_32FC1); 

  //(a*c + b*d)/(c*c + d*d) = re 
  cvMul(x1_imageRe, x2_imageRe, addend1); 
  cvMul(x1_imageIm, x2_imageIm, addend2); 
  cvAdd(addend1, addend2, numerator1); 
  cvMul(x1_imageIm, x1_imageIm, square1); 
  cvMul(x2_imageIm, x2_imageIm, square2); 
  cvAdd(square1, square2, denominator1);
  cvDiv(numerator1, denominator1, x1_imageRe, 1.0 ); 

  //(b*c - a*d)/(c*c + d*d) = im 
  cvMul(x1_imageIm, x2_imageRe, addend1); 
  cvMul(x1_imageRe, x2_imageIm, addend2); 
  cvSub(addend1, addend2, numerator2); 
  cvMul(x1_imageIm, x1_imageIm, square1); 
  cvMul(x2_imageIm, x2_imageIm, square2); 
  cvAdd(square1, square2, denominator2); 
  cvDiv(numerator2, denominator2, x1_imageIm, -1.0 ); 

  //re und im zusammenfügen 
  CvMat *dst1 = cvCreateMat(src1.height, src1.width, CV_32FC2);
  cvMerge( x1_imageRe, x1_imageIm, NULL, NULL, dst1 );
  Mat dst=cvarrToMat(dst1);
  return dst;
} 

void Mosse::DefineGoal(void)
{
  Mat g=Mat::zeros(size,CV_32F);
  g.at<double>(int(w/2),int(h/2))=1;
  GaussianBlur(g, g, Size(-1,-1), 2.0);
  // g/=g.max();
  dft(g,G,DFT_COMPLEX_OUTPUT);
}
void Mosse::PreProcess(Mat &window)
{
    //log(pixel)
	Mat window32;
	window.convertTo(window32,CV_32FC1);
  Mat Dia=Mat::ones(window32.size(),CV_32FC1);
  log(window32+Dia,window);

	//normalize
	Mat mean,StdDev;
  meanStdDev(window,mean,StdDev);
  window=(window-mean.at<double>(0)*Dia)/StdDev.at<double>(0);
  
  //Gaussain weighting
  window=window.mul(HanWin);
}

double Mosse::Correlate(Mat image_sub,Point &delta_xy)
{
  auto IMAGE_sub=image_sub;
  auto RESPONSE=image_sub;
  auto response=image_sub;
  // imshow("img_sub",image_sub);

  //FFT
  dft(image_sub,IMAGE_sub,DFT_COMPLEX_OUTPUT);

  //Performs the per-element multiplication of two Fourier spectrums
  //************************ This is where F*H  ************************
  mulSpectrums(IMAGE_sub, H, RESPONSE, 0, true );

  //inverse FFT
//  idft(RESPONSE, response, DFT_SCALE||DFT_REAL_OUTPUT);
  idft(RESPONSE, response, DFT_SCALE|DFT_REAL_OUTPUT);
  Size s = response.size();
  int height = s.height;
  int width = s.width;

  //find max/min
  double minVal, maxVal;
  Point minLoc, maxLoc;
  minMaxLoc(response, &minVal, &maxVal, &minLoc, &maxLoc);

  //compute x and y
  // delta_xy=Point(maxLoc.x-width/2,maxLoc.y-height/2);
  delta_xy.x=maxLoc.x-width/2;
  delta_xy.x=maxLoc.y-height/2;
  // cout<<"delta_x= "<<delta_xy.x<<", "<<"delta_y= "<<delta_xy.x<<endl;
  
  //compute PSR
  double PSR;
  Mat Mean,Std;
  meanStdDev(response, Mean, Std);
  auto mean=Mean.at<double>(0);
  auto std=Std.at<double>(0);
  PSR=(maxVal-mean)/(std+eps);
  return PSR;
}
float Mosse::randNum(void)
{
    return ((float)rand()) / RAND_MAX;
}
Mat Mosse::randWarp(const Mat& a)
{
  // affine warp matrix
  Mat T= Mat::zeros(2,3,CV_32F);
  
  // random rotation
  double coef=0.2;
  double ang=(randNum()-0.5)*coef; //-0.1~0.1
  double c=cos(ang); double s=sin(ang);
  T.at<float>(0,0) = c + (randNum()-0.5)*coef;
  T.at<float>(0,1) = -s + (randNum()-0.5)*coef;
  T.at<float>(1,0) = s + (randNum()-0.5)*coef;
  T.at<float>(1,1) = c + (randNum()-0.5)*coef;
  
  // random translation
  int h=a.size().height;
  int w=a.size().width;
  Mat center_warp = Mat(2, 1, CV_32F);
  center_warp.at<float>(0,0) = w/2;
  center_warp.at<float>(1,0) = h/2;
  T.col(2) = center_warp - (T.colRange(0, 2))*center_warp;
  
  // do the warpping
  //cout<<T<<endl;
  Mat warped;
  warpAffine(a, warped, T, a.size(), BORDER_REFLECT);
  //cout<<warped<<endl;
  return warped;
}
void Mosse::Run(const Mat &frame)
{
  double x=center.x;
  double y=center.y;

  //get image_sub
  Mat image_sub;
  getRectSubPix(frame, size, center, image_sub); //image_sub is the output array
  
  //preprocess
  PreProcess(image_sub);

  //correlate //Run is the only function call correlate //In correlate, decide delta_xy
  Point delta_xy;
  double PSR=Correlate(image_sub,delta_xy);//use H_i-1
  if(PSR<1)
  {
    std::cout<<"PSR is low :"<<PSR<<std::endl;
  }
  
  //update location
  center.x+=delta_xy.x;
  center.y+=delta_xy.y;


//update filter H_i
  //get image_sub
  getRectSubPix(frame, size, center, image_sub); //image_sub is the output array
  
  //preprocess
  PreProcess(image_sub);

  //update filter
  Mat IMAGE_sub;
  Mat A_new;
  Mat B_new;
  dft(image_sub,IMAGE_sub,DFT_COMPLEX_OUTPUT);
  mulSpectrums(G        , IMAGE_sub, A_new, 0, true );
  mulSpectrums(IMAGE_sub, IMAGE_sub, B_new, 0, true );
  A=A*(1-rate)+A_new*rate;
  B=B*(1-rate)+B_new*rate;
  UpdateFilter();
}
#endif