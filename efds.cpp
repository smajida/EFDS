#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/highgui.h>
#include <iostream>
#include <time.h>
#include "Frame.h"
#include "Queue.h"


using namespace cv;
using namespace std;
// -----------------------------------------------Global Variables------------------------------

//stuff thresholding
int threshold_value = 10, threshold_value_r = 160, threshold_value_g = 25, threshold_value_b = 40; 
int threshold_type = 0;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
int const t_binary = 0;
int const t_binary_inverted = 1;
bool possible_fire = 0; 


Mat cc_mask;

Mat smoke_mask, keyframe, fire_mask;

int mode = 1;   //1 video, 2 camera

int keytimer = 10;


int norm_delay = 3000;
int poss_fire_delay = 1000;
int queueSize=10;
Queue images(queueSize); //create queue of 10 images
FRAME frame, last_frame; 

// Area counter
//int areaCount = 0;

// denoise 
int morph_size = 4;

//test variables
Mat  test, dog, s1, s2, s2subs1, s2subs1_gray, s2subs1_bw, s2subs1_after_morph, strel, back_sub, denoised, segmentedImg, elongTest, image_grab;

// Camera capture stuff
int XthFrame = 20; //XthFrame to be captured after camera turns on
int camera = 1;  //index of camera (in case of multiple cameras
//char* camera;

//Growth Rate variables
float fireGrowthArray[100];
float smokeGrowthArray[100];
int fireGrowthArrayIdx = 0;
int smokeGrowthArrayIdx = 0; 

int vote = 0;

VideoCapture cap;

/** ------------------- Function Headers ---------------------------------------- */
void initialize1(VideoCapture &); 

void addImage1(VideoCapture &);

void initialize2(/*VideoCapture &*/); 

void addImage2(/*VideoCapture &*/);
//void ColorInspection(Mat, Mat &, Mat &, Mat &);

void ColorSep(Mat, Mat &, Mat &, Mat &);

void ColorInspection();

Mat BackgroundSubtraction1(Mat,Mat);

Mat BackgroundSubtraction2(Mat,Mat);

Mat Denoise(Mat, int);

int AreaCounter(Mat);

float elongationFactor(Mat);

void calcGrowthRate(FRAME , FRAME);



void monitorGrowthRate();

void RaiseAlarm();

void sleep(time_t);


void smokeMask();

void fireMask();

void Connector(Mat , int , int );
Mat DoubleThresholding(int, int, Mat, int, int);
Mat equalize(Mat);


// dispersion rate
// edge detection
// vote system?


/** ------------------- Main program ---------------------------------------- */
int main( int argc, char** argv )
{

    //TIMESTAMP STUFF
    time_t timestamp;
    time(&timestamp);
    printf("time %i \n", timestamp);
    struct tm *timeinfo;
    timeinfo = localtime(&timestamp);
    printf("current local = %s",asctime(timeinfo));
    
       
    if(mode == 1){
    
    cap.open(argv[1]);
  
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    initialize1(cap);
    
    }
    
   
    else
     initialize2();
     
    while(true){  //turns camera on, then off, then delays
    
          if(mode ==1){
             addImage1(cap);  //grab frame, extract data, add to queue
             waitKey(1000);
             if(keytimer == 0 && !possible_fire){
                keyframe = frame.image;
                //keyframe = equalize(keyframe);
                keytimer = 10;
                }
             else 
                keytimer--;  
          
          }
          else
              addImage2();
              waitKey(10);
          
          time(&timestamp);
          
          //struct tm *timeinfo;
          timeinfo = localtime(&timestamp);
          printf("current local = %s",asctime(timeinfo));
          
          
          
          //if frame contains smoke or fire (initial test), or if possible fire previously detected
          //and has not been ruled out to be false
          
          if( possible_fire){ 
               //do further testing
               // growth rate calculations
               // dispersion rate calc
               //          
               calcGrowthRate(frame, last_frame);
               cout<<"Vote = "<<vote<<endl;
               
               /*
               if(frame.growthRate > 1.0)
               {
                    RaiseAlarm();
                    //speed up frame grabbing
                    //if(waitKey(norm_delay) >= 0) break;  //delay
               }
               
               */
               
               
               if(vote < -3){  //no fire/smoke
                    possible_fire = 0;
                    XthFrame = 20; 
               }
               
               else if(vote >= 9){  //certain fire/smoke
                    RaiseAlarm();
                    XthFrame = 20; 
                    }
                    
               else{  //maybe fire/smoke, need more testing
                    possible_fire = 1;
                    XthFrame = 20; 
               }
               
               //sleep(5000);
               //if(waitKey(poss_fire_delay) >= 0) break;  //delay
                 
               
          }
          
          else  //no possible fire detected
          
               //sleep(10000);
          
               //if(waitKey(norm_delay) >= 0) break;  //delay
               XthFrame = 20;
               waitKey(1000);
     } //END WHILE
     

     return 0;
}

//fill Queue before any IP
void initialize1(VideoCapture &cap){

   

     for(int i=0; i<queueSize; i++){
           
          for(int i=0; i<=5; i++){  //captures the Xth frame from camera and adds to queue

                cap>>image_grab;  //save frame
                //waitKey(0);
       
                frame.image = image_grab.clone();
                frame.s = Mat::zeros(image_grab.rows, image_grab.cols, CV_8UC1);

                if(i == 5)   //add frame X to queue (first few frames may not be good quality
                     images.Enqueue(frame);  //add frame to queue
                        
	     
         } //end image capture
             
   
    }
     
     
     waitKey(norm_delay);      
     smoke_mask = Mat::zeros(frame.image.rows, frame.image.cols, CV_8UC1); 
     fire_mask = Mat::zeros(frame.image.rows, frame.image.cols, CV_8UC1); 
     keyframe = frame.image; 
    // keyframe =equalize(keyframe);
     Mat cc_mask = Mat::zeros(frame.image.rows, frame.image.cols, CV_8UC1);
          

};

//fill Queue before any IP
void initialize2(/*VideoCapture &cap*/){

    VideoCapture cap(camera);   //start camera
    //cap.set(CV_CAP_PROP_FPS, 1);

     for(int i=0; i<queueSize; i++){
           
          for(int i=0; i<=5; i++){  //captures the Xth frame from camera and adds to queue

                cap>>image_grab;  //save frame
                //waitKey(0);
                frame.image = image_grab.clone();
                frame.s = Mat::zeros(image_grab.rows, image_grab.cols, CV_8UC1);

                if(i == 5)   //add frame X to queue (first few frames may not be good quality
                     images.Enqueue(frame);  //add frame to queue
	     
         } //end image capture
   
    }
     
     cap.release();  //turn off camera
     
     waitKey(norm_delay);  
     
     smoke_mask = Mat::zeros(frame.image.rows, frame.image.cols, CV_8UC1);  
          

};

//Capture frames and add to Queue
void addImage1(VideoCapture &cap){

     ////////Capture Image/////////////////////
   
    
     for(int i=0; i<=XthFrame; i++){  //captures the Xth frame from camera and adds to queue

          cap>>image_grab;  //save frame
          //waitKey(0);
       
         frame.image = image_grab.clone();

          if(i == XthFrame){   //add frame X to queue (first few frames may not be good quality


               ////////////////Data Extraction//////////////
	          
               //ColorSep(frame.image, frame.R, frame.G, frame.B);
               
	        
	          frame.e_factor = 0;
	          
	         // if(!images.IsEmpty()){
	               images.get_Frame(0, last_frame);
	               //frame.BackSub = BackgroundSubtraction1(last_frame.image, frame.image); 
	               
	               //frame.BackSub = Denoise(frame.BackSub); 
	              imshow("LastFrame", last_frame.image); 
	              imshow("CurrentFrame", frame.image);
	             // imshow("BGSUB", frame.BackSub);
	               //waitKey(0);
	               
	               time_t timestamp;
                    time(&timestamp);
                    frame.timestamp = timestamp;
	               
	         // }
	          
	          //ColorInspection();
	          //fireMask();
	          smokeMask();
	          
	          
	          
	          images.Enqueue(frame);  //add frame to queue
	     
	     
          } // end if
          
        
    } //end image capture
   
    
};


//Capture frames and add to Queue
void addImage2(/*VideoCapture &cap*/){

     ////////Capture Image/////////////////////
    VideoCapture cap(camera);   //start camera
    //cap.set(CV_CAP_PROP_FPS,1);
    
    
     for(int i=0; i<=XthFrame; i++){  //captures the Xth frame from camera and adds to queue

          cap>>image_grab;  //save frame
         //waitKey(0);
       
         frame.image = image_grab.clone();

          if(i == XthFrame){   //add frame X to queue (first few frames may not be good quality


               ////////////////Data Extraction//////////////
	          
               //ColorSep(frame.image, frame.R, frame.G, frame.B);
               
	
	          frame.e_factor = 0;
	          //ColorInspection();
	          
	         // if(!images.IsEmpty()){
	               images.get_Frame(0, last_frame);
	               frame.BackSub = BackgroundSubtraction1(last_frame.image, frame.image); 
	               
	               //frame.BackSub = Denoise(frame.BackSub); 
	             //  imshow("LastFrame", last_frame.image); 
	               imshow("CurrentFrame", frame.image);
	             imshow("BGSUB", frame.BackSub);
	              //waitKey(0);
	               
	               
	               time_t timestamp;
                    time(&timestamp);
                    frame.timestamp = timestamp;
	               
	         // }
	          
	          
	          
	          images.Enqueue(frame);  //add frame to queue
	     
          } // end if
          
        
    } //end image capture
   
    cap.release();  //turn off camera
    
};
    

// area counter for white areas in images, used with background subtraction  
int AreaCounter(Mat image){
  int areaCount = 0;

  for(int i=0; i<image.cols; i++)
  {
	for(int j=0; j<image.rows; j++)
	{
	     
		if(image.at<uchar>(j,i) == 255)
		areaCount++; //update counter for image area counter field
		
	}
	
  }
  
  //printf("areaCount=%i", areaCount);
  return areaCount;
};


void ColorSep(Mat image, Mat &image_r, Mat &image_g, Mat &image_b){
	vector<Mat> rgb;
	Mat  image_r_bw,  image_g_bw,  image_b_bw, anded;
  
	split(image, rgb);
	image_r = rgb[2];
	image_g = rgb[1];
	image_b = rgb[0];

	return;
};


void ColorInspection(){

     vector<Mat> rgb, hsv, hls;

     Mat image_r,
     image_g,
     image_b,
     image_h,
     image_s,
     image_v,
     image_h2,
     image_s2,
     image_l,
     anded_smoke,
     anded_fire;

     Mat image_hsv= Mat::zeros(frame.image.rows, frame.image.cols, CV_8UC3);
       cvtColor(frame.image, image_hsv, COLOR_RGB2HSV);
       
       Mat image_hls= Mat::zeros(frame.image.rows, frame.image.cols, CV_8UC3);
       cvtColor(frame.image, image_hls, COLOR_RGB2HLS);
       
       Mat image_sp= Mat::zeros(frame.image.rows, frame.image.cols, CV_8UC1);

       
       
     // spliting into different color channels
     split(frame.image, rgb);
     image_r = rgb[2];
     image_g = rgb[1];
     image_b = rgb[0];

     split(image_hsv, hsv);
     image_v = hsv[2];
     image_s = hsv[1];
     image_h = hsv[0];

     split(image_hls, hls);
     image_l = hls[2];
     image_s2 = hls[1];
     image_h2 = hls[0];
     
     

      // smoke threadholding --- using hsv and hls
      //HSV thresholding

     int t_hue_above = 2;
     int t_hue_below = 11;
     int t_sat_above = 50;
     int t_sat_below = 80;
     int t_val_below = 160;
     int t_val_above = 200;

     Mat image_h_u, image_h_l, image_s_u, image_s_l,image_v_u, image_v_l;

     threshold(image_h, image_h_u, t_hue_above, max_value,t_binary);
     threshold(image_h, image_h_l, t_hue_below, max_value,t_binary_inverted);
     threshold(image_s, image_s_u, t_sat_above, max_value,t_binary);
     threshold(image_s, image_s_l, t_sat_below, max_value,t_binary_inverted);
     threshold(image_v, image_v_u, t_val_below, max_value,t_binary);
     threshold(image_v, image_v_l, t_val_above, max_value,t_binary_inverted);



     Mat and_hsv = image_h_u & image_h_l & image_s_u & image_s_l & image_v_u & image_v_l;

     //HLS thresholding

     int t_hue2_above = 2;
       int t_hue2_below = 11;
       int t_sat2_above = 130;
       int t_sat2_below = 180;
       int t_lum_below = 35;
       int t_lum_above = 90;

     Mat image_h2_u,image_h2_l,image_s2_u,image_s2_l,image_l_u,image_l_l;

     threshold(image_h2, image_h2_u, t_hue2_above, max_value,t_binary);
     threshold(image_h2, image_h2_l, t_hue2_below, max_value,t_binary_inverted);
     threshold(image_s2, image_s2_u, t_sat2_above, max_value,t_binary);
     threshold(image_s2, image_s2_l, t_sat2_below, max_value,t_binary_inverted);
     threshold(image_l, image_l_u, t_lum_below, max_value,t_binary);
     threshold(image_l, image_l_l, t_lum_above, max_value,t_binary_inverted);


     Mat and_hls = image_h2_u & image_h2_l & image_s2_u & image_s2_l & image_l_u & image_l_l;
     anded_smoke = and_hsv & and_hls;
     Denoise(anded_smoke,2);

     smoke_mask = anded_smoke;  //global variable




     //// FIRE THRESHOLDING
     //need to keep tuning
     int threshold_value_fire_r_upper = 260,
     threshold_value_fire_r_lower = 120,
     threshold_value_fire_g_upper = 260,
     threshold_value_fire_g_lower =140,
     threshold_value_fire_b_upper = 260,
     threshold_value_fire_b_lower = 80;

     Mat anded_fire_d;

     Mat image_r_f_u = DoubleThresholding(threshold_value_fire_r_upper, threshold_value_fire_r_upper+60,
                                          image_r, t_binary_inverted, 0);

     Mat image_r_f_l= DoubleThresholding(threshold_value_fire_r_lower, threshold_value_fire_r_lower-40,
                                          image_r, t_binary, 0);

     anded_fire_d = image_r_f_u & image_r_f_l;

     Mat image_g_f_u = DoubleThresholding(threshold_value_fire_g_upper, threshold_value_fire_g_upper+20,
                                             image_g, t_binary_inverted, 0);

     Mat image_g_f_l= DoubleThresholding(threshold_value_fire_g_lower, threshold_value_fire_g_lower-20,
                                             image_g, t_binary, 0);

     anded_fire_d = anded_fire_d & image_g_f_u & image_g_f_l;

     Mat image_b_f_u = DoubleThresholding(threshold_value_fire_b_upper, threshold_value_fire_b_upper+20,image_r, t_binary_inverted, 0);

     Mat image_b_f_l = DoubleThresholding(threshold_value_fire_b_lower, threshold_value_fire_b_lower,image_r, t_binary, 0);


     anded_fire_d = anded_fire_d & image_b_f_u & image_b_f_l;
     fire_mask = anded_fire_d;

return;
};

Mat BackgroundSubtraction1(Mat current,Mat prev) {
	Mat backSub;
	absdiff(current,prev, backSub);
	
	// Convert the image to Gray
	if (backSub.channels() >1){
	
	cvtColor( backSub, backSub, COLOR_RGB2GRAY ); // cvtColor(src, dst, code,dstCn)	
	threshold(backSub, backSub, threshold_value, max_BINARY_value, 0);
	//backSub = Denoise(backSub, 4);
     }
     else{
	//  threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );
	threshold(backSub, backSub, threshold_value, max_BINARY_value, 0);
	//backSub = Denoise(backSub, 4);
     }
	return backSub;

};


Mat BackgroundSubtraction2(Mat current,Mat prev) {
	int high_thresh = 20;
	int low_thresh = 10;
	int denoise_k_size = 3;
	Mat backSub;
	absdiff(current,prev, backSub);
	
	backSub = DoubleThresholding(high_thresh, low_thresh,backSub, t_binary, denoise_k_size);
	
	/*
	if (backSub.channels() >1){
	
		cvtColor( backSub, backSub, COLOR_RGB2GRAY ); // cvtColor(src, dst, code,dstCn)
		backSub = Denoise(backSub, 4);
		threshold(backSub, backSub, threshold_value, max_value, t_binary);
   }
   else{
		threshold( src_gray, dst, threshold_value, max_value,threshold_type );
		threshold(backSub, backSub, threshold_value, max_value, t_binary);
		backSub = Denoise(backSub, 4);
   }
     */
	return backSub;

};


Mat Denoise(Mat img, int morph_size){
	Mat strel = getStructuringElement(2, Size(2*morph_size+1,2*morph_size+1), Point(-1,-1));
// getStructuringElement(shape, ksize, anchor)
// shape 0 = morph_rect  1= morph_ellipse 2 = morph_cross 
// ksize = size of structuring element
// point (-1,-1) means anchor point is at the center

//morphologyEx(src,dst, operation, element) <-- used to call morphological operations
	morphologyEx(img, img, 2, strel);

/*	imshow( "img_after_morph", img); // show image r channel
	waitKey(0);                                          // Wait for a keystroke in the window
	*/
	return img;
};



float elongationFactor(Mat image){
	float colAveLeft, colAveRight,rowAveTop, rowAveBottom, elongationFactor;
		printf("checkpoint2\n");
     //	imshow( "elong", image); // show image r channel
     //	waitKey(0);  
     // colaveleft	

	int n=0;
	 for(int j=0; j<image.rows;j++)
	 {
		int i =0;
		while((image.at<uchar>(j,i) !=255) && (i<image.cols))  //NOTE: j=rows, i=cols
		{
			i++;
		} //end while
		
		
		if(i<image.cols)
		{
				colAveLeft += i;
				n++;
		} // end if
	 } // end for loop i
	 colAveLeft = colAveLeft/n;
	 printf("n=%i \n",n);
 	 printf("colAveLeft =%f \n",colAveLeft);
	
     //rowAveTop
	int k=0;
	for(int u=0; u<image.cols;u++)
	{
		int v=0;
		while((image.at<uchar>(v,u) !=255) && (v<image.rows))  //NOTE: v=rows, u=cols
		{
			v++;
		} //end while
		
		
		if(v<image.rows)
		{
				rowAveTop += v;
				k++;
		} // end if
	} // end for loop i
	 rowAveTop = rowAveTop/k;
	 printf("k=%i \n",k);
	 printf("rowAveTop=%f \n",rowAveTop);

     //colave right
	int m=0;
	for(int y=0; y<image.rows;y++)
	{
		int x;
		x = image.cols;
		while((image.at<uchar>(y,x) !=255) && (x>0))  //NOTE: y=rows, x=cols
		{
			x--;
		} //end while
		
		
		if(x>0)
		{
				colAveRight += x;
				m++;
		} // end if
	} // end for loop i
	 colAveRight = colAveRight/m;
	 printf("m=%i \n",m);
	 printf("colAveRight=%f \n",colAveRight);





     //rowave bottom
	int t=0;
	for(int s=0; s<image.cols;s++)
	{
		int r;
		r= image.rows;
		while((image.at<uchar>(r,s) !=255) && (r>0))  //NOTE: r=rows, s=cols
		{
			r--;
		} //end while
		
		
		if(r>0)
		{
				rowAveBottom += r;
				t++;
		} // end if
	} // end for loop i
	rowAveBottom = rowAveBottom/t;
	printf("t=%i \n",t);
	printf("rowAveBottom=%f \n",rowAveBottom);

	
	elongationFactor= (rowAveBottom - rowAveTop) / (colAveRight-colAveLeft);
	printf("elongationFactor=%f \n",elongationFactor);


 	return elongationFactor; 	
};

void calcGrowthRate(FRAME current, FRAME prev ){
     // get prev.frame
     frame.growthRate = (current.smokePixels - prev.smokePixels )/(current.timestamp - prev.timestamp);
     printf("growth rate = %f \n", frame.growthRate);
     
     monitorGrowthRate();
     
     

	return ;
};

void monitorGrowthRate(){
     if (fireGrowthArrayIdx == 0)
          time_t  startTime = frame.timestamp;
          
     fireGrowthArray[fireGrowthArrayIdx] = frame.growthRate;
     
     
     if( fireGrowthArrayIdx >= 2){
     if( fireGrowthArray[fireGrowthArrayIdx] > 0 &&
           fireGrowthArray[fireGrowthArrayIdx-1] > 0)
               vote = vote + 3;
     else if ( fireGrowthArray[fireGrowthArrayIdx] < 0 &&
                fireGrowthArray[fireGrowthArrayIdx-1] > 0)
                vote = vote - 3;
     else
          vote = vote;
          
     
     }
     fireGrowthArrayIdx++;
     return;
};


void RaiseAlarm(){ 
if (frame.p_fire)
printf("Tyler how do i turn on the red one?\n");
else if (frame.p_smoke)
printf("SMOKE! SMOKE! SMOKE! SMOKE!\n");
else if (frame.p_smoke && frame.p_fire )
printf("Tyler wtf, how did this happen!?\n");

return;

};


void sleep (time_t delay)
{
     time_t timer0, timer1;
     time(&timer0);
     do
     {
          time(&timer1);
     }
     while ((timer1 - timer0)<delay);

};









Mat DoubleThresholding(int thresh_strong, int thresh_weak, Mat image, int thresh_type, int denoise_k_size)
{
	Mat strong, weak;
	// cc_mask will be a global variable so it can be modified 
	// by the Connector function

	
	// Convert the image to Gray
	if (image.channels() >1){
		cvtColor( image, image, COLOR_RGB2GRAY ); // cvtColor(src, dst, code,dstCn)
   }
   
	//  threshold( src_gray, dst, threshold_value, max_value,threshold_type );


	threshold(image, weak, thresh_weak, 50, thresh_type);
	threshold(image, strong, thresh_strong, 50, thresh_type);


//denoise strong image to get rid of small objects and noise
     if(denoise_k_size>0){
	Denoise(strong, denoise_k_size);
	}
	// this will work because even if denoising the strong image losses some 
	// data, the weak image will have captured it and can be used to threshold
	cc_mask = Mat::zeros(frame.image.rows, frame.image.cols, CV_8UC1);
	cc_mask = strong+weak;


	//imshow("after initial thresholding", cc_mask);
	
	
	
//	Mat cc_mask = Mat::zeros(frame.image.rows, frame.image.cols, CV_8UC1);
// ^ dont really need a separate mask for this
// since the str_weak_plus does the job, just duplicating efforts

	for(int i = 0; i<image.cols; i++)
		{
			for(int j =0; j<image.rows; j++)
				{
					if(cc_mask.at<uchar>(j,i)==100)
					{
						cc_mask.at<uchar>(j,i)=255;
						Connector(cc_mask, i-1,j);
						Connector(cc_mask, i+1,j);
						Connector(cc_mask, i,j-1);
						Connector(cc_mask, i,j+1);
					}
				}
		}
		
		
	for(int i = 0; i<frame.image.cols; i++)
		{
			for(int j =0; j<frame.image.rows; j++)
				{
					if(cc_mask.at<uchar>(j,i)!=255)
					{

						cc_mask.at<uchar>(j,i) = 0;
						

					}
				}
		}

//	imshow("after connector cc_mask", cc_mask);
	//waitKey(0);

	return cc_mask.clone();

};





void Connector(Mat cc_mask, int i, int j)
{
	if (i < 0 || i>cc_mask.cols || j<0 || j>cc_mask.rows || cc_mask.at<uchar>(j,i) != 50 )
	{

		return;
	}
	else
	{
		cc_mask.at<uchar>(j,i) = 255;
		Connector(cc_mask, i-1,j);
		Connector(cc_mask, i+1,j);
		Connector(cc_mask, i,j-1);
		Connector(cc_mask, i,j+1);
		return ;
	}
	
	
};

void smokeMask(){

     Mat current, backSub, fire_image, smoke_image, smoke, fire, sub_smoke;
     
     current = frame.image;
     //current = equalize(current);
     
     backSub = BackgroundSubtraction2(current, keyframe);
     //backSub=Denoise(backSub,2);	    
     
     
	ColorInspection();
	
	bitwise_not(smoke_mask, sub_smoke); //remove smoke pixels from fire mask (smoke mask needs to be really good to catch it all
	fire_mask = fire_mask & sub_smoke;
	
	smoke_image = backSub & smoke_mask;		
     fire_image = backSub & fire_mask;  
     frame.smokePixels = AreaCounter(smoke_image);

     if(frame.smokePixels > 40){
	     frame.p_smoke = true;
	     possible_fire = true;
	
     }
     if(frame.firePixels > 40){
	     frame.p_fire = true;
	     possible_fire = true;
     }
      
     imshow("smoke_image", smoke_image);
     imshow("fire_image", fire_image);
     imshow("Key image", keyframe);
     imshow("BG", backSub);
     
     waitKey(0);
     
     

};

Mat equalize(Mat img){
   vector<Mat> channels; 
       Mat img_hist_equalized;

       cvtColor(img, img_hist_equalized, COLOR_BGR2YCrCb); //change the color image from BGR to YCrCb format

       split(img_hist_equalized,channels); //split the image into channels

       equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)

   merge(channels,img_hist_equalized); //merge 3 channels including the modified 1st channel into one image

      cvtColor(img_hist_equalized, img_hist_equalized, COLOR_YCrCb2BGR); 
      return img_hist_equalized;
      
      };
