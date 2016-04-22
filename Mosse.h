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
  double eps=0.00001;
  double rate=0.125; //learning rate
  double psrThre=6.0;
  const double InRangeParameter=0.1;
  int MaxIteration=3;
  Point_<double> center; //center of the bounding box
  Size size; //size of the bounding box
  Mat previousWin;
  Mat previousRes;
  Mat HanWin;
  
  //Optimal size for DFT processing  
  int h=0;
  int w=0;

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
  void Draw(Mat &frame,bool flag);
  
  //computing
  void Initialize(const Mat &frame, const Rect &rect);
  void DefineGoal(void); 
  void PreProcess(Mat &window); 
  double Correlate(const Mat &image_sub, Point &delta_xy);
  void UpdateFilter(void);
  Mat randWarp(const Mat &a);
  float randNum(void);
  Mat divDFTs(const CvMat &src1, const CvMat &src2);
  // Mat divDFTs(Mat& first, Mat& second);
  Mat addComplexPlane(Mat real);
  bool InRange(const Point &delta_xy);  
  bool Run(const Mat &frame);
  

};
Mosse::Mosse(void)
{
}
Mosse::~Mosse(void)
{
}
void Mosse::Draw(Mat &frame,bool flag)
{
  double x=center.x;
  double y=center.y;
  double x1=int(x-0.5*w); 
  double y1=int(y-0.5*h); 
  double x2=int(x+0.5*w); 
  double y2=int(y+0.5*h);
  Scalar color;
  if(flag==1)
  {
    color=CV_RGB(0,255,0);
  }
  else if(flag==0)
  {
    color=CV_RGB(255,0,0);
  }    
  rectangle(frame,Point(x1,y1),Point(x2,y2), color, 3, 8, 0);
  circle(frame, center, 2,color,-1);
  // imshow("Tracking",frame);
}
void Mosse::Initialize(const Mat &frame, const Rect &rect)
{
  // CleanUp();
  //Get the optimal size for DFT processing
  w=getOptimalDFTSize(rect.width);
  h=getOptimalDFTSize(rect.height);
  
  //Get the center position
  int x1=floor((2*rect.x+rect.width-w)/2);
  int y1=floor((2*rect.y+rect.height-h)/2);
  // center.x=x1+(w-1)/2;//
  // center.y=y1+(h-1)/2;//
  center.x=x1+(w)/2;//
  center.y=y1+(h)/2;//
  size.width=w;
  size.height=h;
  
  //Initialize FFT
  Mat window;
  getRectSubPix(frame, size, center, window); //window is the output array

  //create Hanning window
  createHanningWindow(HanWin, size, CV_32F);
  // imshow("HanWin",HanWin);

  //define G
  DefineGoal();
  
  //compute A,B and H
  //affine image
  A=Mat::zeros(G.size(), G.type()); // A.size()=B.size()=G.size()
  B=Mat::zeros(G.size(), G.type()); // A.size()=B.size()=G.size()
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
    A+=A_i;
    B+=B_i;
  }


  //update filter
  UpdateFilter();
  Run(frame);
  // imshow("Template",window);
  std::cout<<"Initialization completed"<<std::endl;
}

void Mosse::UpdateFilter(void)
{
  H=divDFTs(A,B);
  // H*=-1;
  // cout<<"********"<<H<<"********"<<endl;
}

Mat Mosse::divDFTs(const CvMat &src1, const CvMat &src2)
{ 
  /*
    Element-wise division of complex numbers in src1 and src2
  */ 
  
  CvMat* src1_Re = cvCreateMat(size.height, size.width, CV_32FC1);
  CvMat* src1_Im = cvCreateMat(size.height, size.width, CV_32FC1);
  CvMat* src2_Re = cvCreateMat(size.height, size.width, CV_32FC1);
  CvMat* src2_Im = cvCreateMat(size.height, size.width, CV_32FC1);
  //CvMat* imageImRot = cvCreateMat(this->source_depth_dft->height,this->source_depth_dft->width, CV_32FC1); 
  cvSplit(&src1, src1_Re, src1_Im, 0, 0 ); 
  cvSplit(&src2, src2_Re, src2_Im, 0, 0 );

  // Divide real and imaginary components 
  CvMat* denominator1 = cvCreateMat(size.height, size.width, CV_32FC1);
  CvMat* denominator2 = cvCreateMat(size.height, size.width, CV_32FC1); 
  CvMat* numerator1 = cvCreateMat(size.height, size.width, CV_32FC1);
  CvMat* numerator2 = cvCreateMat(size.height, size.width, CV_32FC1); 
  CvMat* addend1 = cvCreateMat(size.height, size.width, CV_32FC1); 
  CvMat* addend2 = cvCreateMat(size.height, size.width, CV_32FC1); 
  CvMat* square1 = cvCreateMat(size.height, size.width, CV_32FC1); 
  CvMat* square2 = cvCreateMat(size.height, size.width, CV_32FC1); 

 // (Re1*Re2 + Im1*Im1)/(Re2*Re2 + Im2*Im2) = Re
  cvMul(src1_Re, src2_Re, addend1); 
  cvMul(src1_Im, src2_Im, addend2); 
  cvAdd(addend1, addend2, numerator1); 
  cvMul(src2_Re, src2_Re, square1);
  cvMul(src2_Im, src2_Im, square2); 
  cvAdd(square1, square2, denominator1);
  cvDiv(numerator1, denominator1, src1_Re, 1.0 ); 

  // (Im1*Re2 - Re1*Im2)/(Re2*Re2 + Im2*Im2) = Im
  cvMul(src1_Im, src2_Re, addend1); 
  cvMul(src1_Re, src2_Im, addend2); 
  cvSub(addend1, addend2, numerator2); 
  cvMul(src2_Re, src2_Re, square1);
  cvMul(src2_Im, src2_Im, square2); 
  cvAdd(square1, square2, denominator2); 
  cvDiv(numerator2, denominator2, src1_Im, -1.0 ); // Complex conjugate

  // Merge Re and Im back into a complex matrix
  CvMat* dst1 = cvCreateMat(src1.height, src1.width, CV_32FC2);
  cvMerge( src1_Re, src1_Im, NULL, NULL, dst1 );
  Mat dst = cvarrToMat(dst1);
  return dst;
} 
void Mosse::DefineGoal(void)
{
  Mat g=Mat::zeros(size,CV_32F);
  g.at<float>(h/2,w/2)=1;
  GaussianBlur(g,g, Size(-1,-1), 2.0);
  double minVal; double maxVal;
  minMaxLoc( g, &minVal, &maxVal);
  g=g/maxVal;
  // imshow("g",g);
  dft(g,G,DFT_COMPLEX_OUTPUT);
}
void Mosse::PreProcess(Mat &window)
{
  window.convertTo(window,CV_32FC1);
  Mat Dia=Mat::ones(window.size(),window.type());
  log(window+Dia,window);
	
  //normalize
  Mat mean,StdDev;
  meanStdDev(window,mean,StdDev);
   window=(window-mean.at<double>(0)*Dia)/StdDev.at<double>(0);
//  window=(window-mean)/(StdDev+eps);
  
  //Gaussain weighting
  window=window.mul(HanWin);
  // imshow("PreProcesswindow",window);
}

double Mosse::Correlate(const Mat &image_sub,Point &delta_xy)
{
  Mat IMAGE_SUB;
  Mat RESPONSE;
  Mat response;
  // imshow("img_sub",image_sub);

  //FFT
  dft(image_sub,IMAGE_SUB,DFT_COMPLEX_OUTPUT);

  //Performs the per-element multiplication of two Fourier spectrums
  //************************ This is where F*H  ************************
  mulSpectrums(IMAGE_SUB, H, RESPONSE, 0, true );

  //inverse FFT
  idft(RESPONSE, response, DFT_SCALE|DFT_REAL_OUTPUT);

  //find max/min
  double minVal, maxVal;
  Point minLoc, maxLoc;
  minMaxLoc(response, &minVal, &maxVal, &minLoc, &maxLoc);

  //compute x and y
  // delta_xy=Point(maxLoc.x-width/2,maxLoc.y-height/2);
  //  cout<<"maxLoc.x= "<<maxLoc.x<<" maxLoc.y= "<<maxLoc.y<<endl;
  delta_xy.x=maxLoc.x-int(response.size().width/2);
  delta_xy.y=maxLoc.y-int(response.size().height/2);
  // delta_xy.x=maxLoc.x;
  // delta_xy.y=maxLoc.y;

  // cout<<"delta_x= "<<delta_xy.x<<", "<<"delta_y= "<<delta_xy.y<<endl;
  
  //compute PSR
  double PSR;
  Mat Mean,Std;
  meanStdDev(response, Mean, Std);
  auto mean=Mean.at<double>(0);
  auto std=Std.at<double>(0);
  PSR=(maxVal-mean)/(std+eps);
  // cout<<"PSR= "<<PSR<<endl;
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
bool Mosse::InRange(const Point &delta_xy)
{
  if(abs(delta_xy.x)<w*InRangeParameter && abs(delta_xy.y)<h*InRangeParameter)
  {
    return true;
  }
  return false;
}

bool Mosse::Run(const Mat &frame)
{
  bool NearCenter=false;
  double PSR=0;
  int iteration=0;
  while(1)
  {
    //get image_sub
    Mat image_sub;
    getRectSubPix(frame, size, center, image_sub); //image_sub is the output array
    //preprocess
    PreProcess(image_sub);
    // imshow("img_sub",image_sub);

    //correlate //Run is the only function call correlate //In correlate, decide delta_xy
    Point delta_xy;
    PSR=Correlate(image_sub,delta_xy);//use H_i-1

    NearCenter=InRange(delta_xy);
    if(NearCenter==false)
    {
       cout<<"Fast Tracking"<<endl;
    }
    //update location
    center.x+=delta_xy.x;
    center.y+=delta_xy.y;
    if(NearCenter || iteration>MaxIteration)
    {
      break;
    }
    iteration++;
  }

  if(PSR<psrThre)
  {
     std::cout<<"PSR is low :"<<PSR<<std::endl;
     return false;
  }
  else
  {
//    std::cout<<"PSR= "<<PSR<<std::endl;
  }

//update filter H_i
  //get image_sub
  Mat img_sub_new;
  getRectSubPix(frame, size, center, img_sub_new);
  
  //preprocess
  PreProcess(img_sub_new);

  //update filter
  Mat F;
  Mat A_new;
  Mat B_new;
  dft(img_sub_new,F,DFT_COMPLEX_OUTPUT);
  mulSpectrums(G, F, A_new, 0, true );
  mulSpectrums(F, F, B_new, 0, true );
  A=A*(1-rate)+A_new*rate;
  B=B*(1-rate)+B_new*rate;
  UpdateFilter();
  return true;
}
#endif