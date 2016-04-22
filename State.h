#ifndef State_IS_INCLUDED
#define State_IS_INCLUDED
class State
{
  public:
    Mat frame;
    Mat frame_gray;
    Rect rect;
    Rect rect_template;
    bool Select_Completed=0;
    bool select_flag = 0;
    Point point1;
    Point point2;
    bool draw_flag=0;
    int drag = 0;
    static void mouseHandler(int event, int x, int y, int flags, void *param);
};
// void State::mouseHandler(int event, int x, int y, int flags, void *param)
// {
//     if (event == CV_EVENT_LBUTTONDOWN && !drag)
//     {
//         // /* left button clicked. ROI selection begins */
//         point1 = Point(x, y);
//         drag = 1;
//         cout<<"mouse down"<<endl;
//     }
//     if (event == CV_EVENT_MOUSEMOVE && drag)
//     {
//         draw_flag=1;
//         point2 = Point(x, y);
//     }
//     if (event == CV_EVENT_LBUTTONUP && drag)
//     {
//         point2 = Point(x, y);
//         rect_template = Rect(point1.x, point1.y, x - point1.x, y - point1.y);//should not include the red box lines
//         drag = 0;
//         Mat img2;
//         // CameraFeed.copyTo(img2);
//         // box = img2(rect_template);
//         cout<<"mouse up"<<endl;
//     }
//     if (event == CV_EVENT_LBUTTONUP)
//     {
//         select_flag = 1;
//         drag = 0;
//         draw_flag=0;
//     }
// }
#endif