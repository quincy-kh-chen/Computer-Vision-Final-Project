/*
 16-720 project
 Purpose: main function
 
 @author Quincy Chen
 email kunhsinc@andrew.cmu.edu
 @version 1 4/10/16
*/

#include <sstream>
#include <string>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include "Mosse.h"
using namespace std;
using namespace cv;

//for mouse motion
Point point1, point2; // vertical points of the bounding box
int drag = 0;
Rect rect; // bounding box
int select_flag = 0;
//for draing selecting box
int draw_flag=0;

//Matrix to store each frame of the webcam feed
Mat CameraFeed;
Mat frame;
Mat box; //box - the part of the image in the bounding box

//video capture object to acquire webcam feed
VideoCapture capture; //********videocapture is a class********

//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
//const int MAX_NUM_OBJECTS=50;
////minimum and maximum object area
//const int MIN_OBJECT_AREA = 20*20;
//const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;


string intToString(int number)
{
    std::stringstream ss;
    ss << number;
    return ss.str();
}


void mouseHandler(int event, int x, int y, int flags, void *param)
{
    if (event == CV_EVENT_LBUTTONDOWN && !drag)
    {
        /* left button clicked. ROI selection begins */
        point1 = Point(x, y);
        drag = 1;
        cout<<"mouse down"<<endl;
    }
    if (event == CV_EVENT_MOUSEMOVE && drag)
    {
        draw_flag=1;
        point2 = Point(x, y);
    }
    if (event == CV_EVENT_LBUTTONUP && drag)
    {
        point2 = Point(x, y);
        rect = Rect(point1.x, point1.y, x - point1.x, y - point1.y);//should not include the red bo lines
        // rect = Rect(point1.x+2, point1.y+2, x - point1.x-4, y - point1.y-4);//should not include the red bo lines
        drag = 0;
        Mat img2;
        CameraFeed.copyTo(img2);
        box = img2(rect);
    }
    if (event == CV_EVENT_LBUTTONUP)
    {
        select_flag = 1;
        drag = 0;
        draw_flag=0;
    }
}


int main(int argc, char* argv[])
{
    rect.width=0; rect.height=0;
    
    //set height and width of capture frame
    capture.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);//FRAME_WIDTH = 640;
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);//FRAME_HEIGHT = 480;
    
    //open capture object at location zero (default location for webcam)
    capture.open(0);
//    capture.open("L.mp4");
    if(!capture.isOpened())
    {
        cout<<"Webcam can't be opened"<<endl;
        getchar();
        return -1;
    }


    //create tracker
    Mosse tracker;
    
    //store image to matrix //cameraFeed is a matrix
    capture.read(CameraFeed);
    //show frames
    imshow("Orginal pic",CameraFeed);//original pic
    setMouseCallback("Orginal pic", mouseHandler, NULL);
 
    Mat showbox;
    while(!select_flag)
    {
        capture.read(CameraFeed);
        CameraFeed.copyTo(showbox);
        if(draw_flag==1)
        {
            rectangle(showbox, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);
        }
        //delay 20ms so that screen can refresh.
        imshow("Orginal pic",showbox);//original pic
        //image will not appear without this waitKey() command
        if(waitKey(10)==27)
        {
            break;
        }
    }
    
    imshow("Template", box);
    cvtColor(CameraFeed,frame,CV_RGB2GRAY);
    tracker.Initialize(frame, rect);
    cout<<"w= "<<tracker.w<<", "<<"h= "<<tracker.h<<endl;
    cout<<"Initial x= "<<tracker.center.x<<", "<<"Initial y= "<<tracker.center.y<<endl;
    
    cout<<"start tracking"<<endl;
    while(select_flag)
    {   
        capture.read(CameraFeed);
        cvtColor(CameraFeed,frame,CV_RGB2GRAY);
        tracker.Run(frame);
        tracker.Draw(CameraFeed);
//        cout<<"x= "<<tracker.center.x<<", "<<"y= "<<tracker.center.y<<endl;
        
        if(waitKey(10)==27)
        {
            break;
        }
    }
    
    
    
    return 0;
}
