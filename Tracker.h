#ifndef Tracker_IS_INCLUDED
#define Tracker_IS_INCLUDED
#include "Mosse.h"
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <opencv/cv.h>
// #include "State.h"
using namespace cv;
class Tracker
{
public:
	Mosse tracker_mosse;
	//for mouse motion
	Point point1, point2; // vertical points of the bounding box
	int drag = 0;
	Rect rect; // bounding box
	int select_flag = 0;
	int init_flag=0;
	//for draing selecting box
	// int draw_flag=0;
    Tracker(void);
    Tracker(const Mat &frame_gray, const Rect &rect);
    ~Tracker(void);
	void SelectTemplate(Mat &frame,const Point &point1,const Point &point2, bool &draw_flag);
	void Initialize(const Mat &frame_gray, const Rect &rect);
	void Update(Mat &frame, const Mat &frame_gray);
	void Run(Mat &frame, const Mat &frame_gray, const Rect &rect, bool &select_flag,const Point &point1,const Point &point2, bool &draw_flag);
};
Tracker::Tracker(void)
{
}
Tracker::Tracker(const Mat &frame_gray, const Rect &rect)
{
  select_flag=1;
  Initialize(frame_gray,rect);
}
Tracker::~Tracker(void)
{
}
void Tracker::SelectTemplate(Mat &frame,const Point &point1,const Point &point2,bool &draw_flag)
{
  if(draw_flag==1)
  {
	  rectangle(frame, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);
  }
}
void Tracker::Initialize(const Mat &frame_gray, const Rect &rect)
{
  tracker_mosse.Initialize(frame_gray, rect);
  init_flag=1;
}
void Tracker::Update(Mat &frame, const Mat &frame_gray)
{
  bool flag=tracker_mosse.Run(frame_gray);
  tracker_mosse.Draw(frame,flag);
}
void Tracker::Run(Mat &frame, const Mat &frame_gray, const Rect &rect, bool &Select_Completed,const Point &point1,const Point &point2, bool &draw_flag)
{
	// cout<<"tracker is running"<<endl;
	if(!select_flag)
	{
		SelectTemplate(frame,point1,point2,draw_flag);
		if(Select_Completed==1)
		{
			select_flag=1;
		}
	}
	if(select_flag && !init_flag)
	{
		Initialize(frame_gray,rect);
	}
	if(select_flag && init_flag)
	{
		Update(frame,frame_gray);
	}

}

#endif