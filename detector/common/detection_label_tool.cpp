/*
一个用于detection label的标注工具，用于对某个指定文件夹下所有图片进行画框标注，并将标注结果存储于指定
文件夹内，标注格式是[x,y,w,h]

使用方法：
1. 运行该软件
2. 按住鼠标左键拖动，会出现一个绿色矩形框，放开鼠标左键即标注结束。
3. 若本次矩形框标注不好，可以按鼠标右键进行撤销本次矩形框标注；若本次矩形框标注正确，则按'a'键将本次标注加入
正式的标注候选集合。
4. 若一张图所有矩形框都画完后，可以按'n'键，对正式标注候选集合所有标注写入文件保存，并自动跳到下一张待标注图像。

本工具在Windows VS2013下测试运行，应该在Linux上也可顺利运行。
*/



#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include<io.h>


using namespace cv;
using namespace std;

Point ptL, ptR; //鼠标画出矩形框的起点和终点,矩形的左下角和右下角
Mat imageSource, imageSourceCopy;


struct UserData
{
    Mat src;
    vector<Rect> rect;
};

void getFiles(string path, vector<string>& files)
{
    //文件句柄
    long   hFile = 0;
    //文件信息
    struct _finddata_t fileinfo;
    string p;
    if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1)
    {
        do
        {
            //如果是目录,迭代之
            //如果不是,加入列表
            if ((fileinfo.attrib &  _A_SUBDIR))
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    getFiles(p.assign(path).append("/").append(fileinfo.name), files);
            }
            else
            {
                files.push_back(fileinfo.name);
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}


void OnMouse(int event, int x, int y, int flag, void *dp)
{
    UserData *d = (UserData *)dp;
    imageSourceCopy = imageSource.clone();

    if (event == CV_EVENT_LBUTTONDOWN)  //按下鼠标右键，即拖动开始
    {
        ptL = Point(x, y);
        ptR = Point(x, y);
    }
    if (flag == CV_EVENT_FLAG_LBUTTON)   //拖拽鼠标右键，即拖动进行
    {
        ptR = Point(x, y);
        imageSourceCopy = imageSource.clone();
        rectangle(imageSourceCopy, ptL, ptR, Scalar(0, 255, 0));
        imshow("标注", imageSourceCopy);

    }
    if (event == CV_EVENT_LBUTTONUP)  //拖动结束
    {
        if (ptL != ptR)
        {
            rectangle(imageSourceCopy, ptL, ptR, Scalar(0, 255, 0));
            imshow("标注", imageSourceCopy);

            int h = ptR.y - ptL.y;
            int w = ptR.x - ptL.x;


            printf("选择的信息区域是:x:%d  y:%d  w:%d  h:%d\n", ptL.x, ptL.y, w, h);

            d->rect.push_back(Rect(ptL.x, ptL.y, w, h));
            //d->src(imageSourceCopy);
        }
    }

    //点击右键删除一个矩形
    if (event == CV_EVENT_RBUTTONDOWN)
    {
        if (d->rect.size() > 0)
        {
            Rect temp = d->rect.back();

            printf("删除的信息区域是:x:%d  y:%d  w:%d  h:%d\n", temp.x, temp.y, temp.width, temp.height);
            d->rect.pop_back();

            for (int i = 0; i < d->rect.size(); i++)
            {
                rectangle(imageSourceCopy, d->rect[i], Scalar(0, 255, 0), 1);
            }

        }
    }

}


void DrawArea(string file_name, string in_path, string out_path)
{
    string img_full_path = in_path + "/" + file_name;
    cout << img_full_path << endl;
    Mat src = imread(img_full_path);
    string pureName = file_name.substr(0, file_name.rfind("."));
    string label_full_path = out_path + "/" + pureName + ".txt";

    FILE* fp = fopen(label_full_path.c_str(), "w+");

    Mat img = src.clone();
    char c = 'x';
    UserData d;
    d.src = img.clone();
    while (c != 'n')
    {
        Mat backup = src.clone();
        imageSource = img.clone();

        namedWindow("标注", 1);
        imshow("标注", imageSource);
        setMouseCallback("标注", OnMouse, &d);

        c = waitKey(0);

        if (c == 'a')
        {
            printf("rect size: %d\n", d.rect.size());
            for (int i = 0; i < d.rect.size(); i++)
            {
                rectangle(backup, d.rect[i], Scalar(0, 255, 0), 1);
            }

            img = backup.clone();

        }
    }


    for (int i = 0; i < d.rect.size(); i++)
    {
        Rect t = d.rect[i];

        fprintf(fp, "%d,%d,%d,%d\n", t.x, t.y, t.width, t.height);
    }

    fclose(fp);


}
int main()
{
    vector<string> files;
    string root_in = "C:/Users/Administrator/Desktop/extra_images";
    string root_out = "C:/Users/Administrator/Desktop/extra_labels";

    ////获取该路径下的所有文件
    getFiles(root_in, files);

    int size = files.size();
    for (int i = 0; i < size; i++)
    {
        DrawArea(files[i], root_in, root_out);
    }

    return 0;
}