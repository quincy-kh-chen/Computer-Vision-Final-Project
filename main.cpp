/*
 16-720 project
 Purpose: main function
 
 @author Quincy Chen
 email kunhsinc@andrew.cmu.edu
 @version 1 4/10/16
*/
#include <iostream>
#include <stdio.h>

#include <fstream>
#include <sstream> 
#include <string> 
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <time.h>
#include "Tracker.h"
// #include "State.h"

using namespace std;
using namespace cv;

//Multiple objects tracking
vector<Tracker> Trackers;

//for mouse motion
Point point1, point2; // vertical points of the bounding box
int drag = 0;
Rect rect_template; // bounding box
bool select_flag = 0;

//for draing selecting box
bool draw_flag=0;

//Matrix to store each frame of the webcam feed
Mat cameraFeed;
Mat frame;
Mat box; //box - the part of the image in the bounding box

//video capture object to acquire webcam feed
VideoCapture capture; //********videocapture is a class********
int waitKeyTime;

//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

void mouseHandler(int event, int x, int y, int flags, void *param)
{
    if (event == CV_EVENT_LBUTTONDOWN && !drag)
    {
        // /* left button clicked. ROI selection begins */
        point1 = Point(x, y);
        drag = 1;
        cout << "mouse down" << endl;
    }
    if (event == CV_EVENT_MOUSEMOVE && drag)
    {
        draw_flag = 1;
        point2 = Point(x, y);
    }
    if (event == CV_EVENT_LBUTTONUP && drag)
    {
        point2 = Point(x, y);
        rect_template = Rect(point1.x, point1.y, x - point1.x, y - point1.y); //should not include the red box lines
        drag = 0;
        Mat img2;
        cameraFeed.copyTo(img2);
        box = img2(rect_template);
        cout << "mouse up" << endl;
    }
    if (event == CV_EVENT_LBUTTONUP)
    {
        select_flag = 1;
        drag = 0;
        draw_flag = 0;
    }
}
string intToString(int number)
{
    std::stringstream ss;
    ss << number;
    return ss.str();
}
void ReadGroundTruth(const string pathToGroundTruth, Rect &rect_template)
{
    //Each row in the ground-truth files: (x, y, box-width, box-height).
    int x, y;
    int box_width, box_height;
    
//    const char * c = pathToGroundTruth.c_str();
//    FILE *fp = fopen(c, "r");
//    fscanf(fp, "%d,%d,%d,%d \n", &x, &y, &box_width, &box_height);
    
    string line;
    ifstream infile;
    infile.open(pathToGroundTruth);
    // while(!infile.eof) 


        std::getline(infile, line); // Saves the line in line
        //remove comma
        
        std::string::size_type n = 0;
        while ( (n = line.find(',', n)) != std::string::npos )
        {
            line.replace( n, 1, 1, ' ' );
            n++;
        }
        //remove com
    
        istringstream in(line); //make a stream for the line itself
    
        int c = in.peek();  // peek character
        if(isdigit(c)|| c=='-')
        {
            in >> x >> y >> box_width >> box_height;

            cout<< "x = " << x << "y = " << y << "w = " << box_width << "h = " << box_height << endl;
        }
    
    rect_template = Rect(x, y, box_width, box_height);
    infile.close();
}

int main(int argc, char* argv[])
{    
    bool ImageSequence = 0;

    if (ImageSequence)
    {
        string pathToData("/Users/WeiHsin/Desktop/TestVideo");
        string pathToVideo("/Matrix");
        string pathToImg = pathToData+pathToVideo + "/img/%04d.jpg";
        string pathToGroundTruth = pathToData+pathToVideo + "/groundtruth_rect.txt";
        ReadGroundTruth(pathToGroundTruth, rect_template);
        capture.open(pathToImg);
        if (!capture.isOpened())
        {
            cout << "Can't not read image sequence" << endl;
            getchar();
            return -1;
        }
        //set height and width of capture frame
        capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); //FRAME_WIDTH = 640;
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);//FRAME_HEIGHT = 480;
        capture.set(CV_CAP_PROP_FPS, 2);

//        ReadGroundTruth(pathToGroundTruth, rect_template);
        //store image to matrix //cameraFeed is a matrix
        capture.read(cameraFeed);
        if (cameraFeed.channels() == 3)
        {
            cvtColor(cameraFeed, frame, CV_RGB2GRAY);
        
        }
        if (cameraFeed.channels() == 1)
        {
            frame = cameraFeed;
            
        }
        // rect_template = Rect(x, y, box_width, box_height);

        Tracker track_mosse(frame, rect_template);
        Trackers.push_back(track_mosse);

        imshow("Tracking", cameraFeed); //original pic
        select_flag = 1;
        waitKeyTime = 10;
    }

    else if (!ImageSequence)
    {
        //open capture object at location zero (default location for webcam)
        capture.open(0);
        if (!capture.isOpened())
        {
            cout << "Webcam can't be opened" << endl;
            getchar();
            return -1;
        }
        capture.set(CV_CAP_PROP_FPS, 30);
        capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); //FRAME_WIDTH = 640;
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT); //FRAME_HEIGHT = 480;

        Tracker track_mosse;
        Trackers.push_back(track_mosse);
        //store image to matrix //cameraFeed is a matrix
        capture.read(cameraFeed);
        imshow("Tracking", cameraFeed);//original pic
        select_flag = 0;
        waitKeyTime = 1;
    }


    
    
 Start: 
    while (1)
    {
        if (!ImageSequence)
        {
            setMouseCallback("Tracking", mouseHandler, NULL);
        }
        clock_t begin_time = clock();

        capture.read(cameraFeed);
        if (cameraFeed.channels() == 3)
        {
            cvtColor(cameraFeed, frame, CV_RGB2GRAY);
            
        }
        if (cameraFeed.channels() == 1)
        {
            frame = cameraFeed;
            
        }

        if (frame.size().height == 0)
        {
            cout << "image sequence end" << endl;
            getchar();
            return -1;
        }

        double CameraTime = float(clock() - begin_time)*1000 / CLOCKS_PER_SEC;
        //cout << "Camera taken: "<<CameraTime<<" ms, "<<" ";

/////////////////////////////////////////////////////////////////////////////////////////////////
        begin_time = clock();

        for (int i = 0; i < Trackers.size(); i++)
        {
            Trackers[i].Run(cameraFeed, frame, rect_template, select_flag, point1, point2, draw_flag);
            
            putText(cameraFeed, "w" + intToString(i+1) + ": " + intToString(Trackers[i].tracker_mosse.w),
                    Point(FRAME_WIDTH-160, 50+i*20), 5, 1, Scalar(255, 0, 0), 1);
            
            putText(cameraFeed, "h" + intToString(i+1) + ": " + intToString(Trackers[i].tracker_mosse.h),
                    Point(FRAME_WIDTH-100, 50+i*20), 5, 1, Scalar(255, 0, 0), 1);

        }

        float ElapsedTime = float(clock() - begin_time)*1000 / CLOCKS_PER_SEC;
        // cout << "Time taken: "<<ElapsedTime<<" ms, "<<"fps : "<<int(1000/ElapsedTime)<<endl;
        // putText(CameraFeed,"w:"+intToString(tracker.w),Point(FRAME_WIDTH-140,50),5,1,Scalar(255,0,0),1);
        // putText(CameraFeed," h:"+intToString(tracker.h),Point(FRAME_WIDTH-80,50),5,1,Scalar(255,0,0),1);
        if (select_flag==0)
        {
            putText(cameraFeed, " Selecting Template!", Point(0, 20), 5, 1, CV_RGB(0, 0, 255), 2);
        }
       
        putText(cameraFeed, "fps: " + intToString(int(1000/ElapsedTime)), Point(FRAME_WIDTH-160, 20),
                5, 1, Scalar(255, 0, 0), 1);
        imshow("Tracking",cameraFeed);
/////////////////////////////////////////////////////////////////////////////////////////////////
        
        if (waitKey(waitKeyTime) == 13)
        {
            cout << "New Object!" << endl;
            select_flag = 0;
            Tracker track_mosse;
            Trackers.push_back(track_mosse);
        }        
        
        if (waitKey(waitKeyTime) == 32)
        {
            select_flag = 0;
            Trackers.clear();
            goto Start;
        }
        if (waitKey(waitKeyTime) == 27)
        {
            break;
        }

    }
    
    return 0;
}
