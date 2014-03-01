#ifndef FRAME_H
#define FRAME_H
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/highgui.h>
#include <time.h>

using namespace std;
using namespace cv;

struct FRAME{
 
  int index;
  time_t timestamp;
  bool smokeOrFire, p_fire, p_smoke;	
  int smokePixels;
  int firePixels;
  Mat image,andedFire, andedSmoke, BackSub, s;
  float e_factor;
  bool edge;
  float growthRate;
      
};

#endif
