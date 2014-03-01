#include "Queue.h"
#include <iostream>
#include <ostream>
#include <iomanip>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/highgui.h>

using namespace std;

//******************************************************************** 
// METHOD NAME: Queue
// METHOD PURPOSE: Constructor
// INPUT PARAMETERS: 
// size - size of queue
// RETURN VALUE � no return value
//********************************************************************

Queue::Queue(int size)
{
  queueArray = new FRAME[size];
  queueSize = size;
  oldest = -1;
  newest = -1;
  n = 0;
}

//******************************************************************** 
// METHOD NAME: Queue
// METHOD PURPOSE: Copy Constructor
// INPUT PARAMETERS: 
// obj - queue from which data is to be copied from onto a new queue
// RETURN VALUE � no return value
//********************************************************************

Queue::Queue(const Queue &obj)
{
  queueArray = new FRAME[obj.queueSize];
  queueSize = obj.queueSize;
  oldest = obj.oldest;
  newest = obj.newest;
  n = obj.n;
  for(int i=0; i < obj.queueSize; i++)
    {
      queueArray[i] = obj.queueArray[i];
    }
}

//******************************************************************** 
// METHOD NAME: ~Queue
// METHOD PURPOSE: Destructor
// INPUT PARAMETERS: 
// RETURN VALUE � no return value
//********************************************************************
Queue::~Queue()
{
  oldest = -1;
  newest = -1;
  n = 0;
  delete [] queueArray;
}


//******************************************************************** 
// METHOD NAME: Enqueue
// METHOD PURPOSE: Add data to queue
// INPUT PARAMETERS: 
// data: of struct STUDENT holding data to be added to queue
//RETURN VALUE � success
//********************************************************************
void Queue::Enqueue(FRAME data)
{
  if(IsFull()){
    newest = (newest + 1) % queueSize;
    oldest = (oldest + 1) % queueSize;
    queueArray[newest] = data;
    cout<<"Frame replaced image in slot "<<newest<<endl;
    
  }
  else{
    newest = (newest + 1) % queueSize;
    queueArray[newest] = data;
    n++;
    cout<<"Frame added to slot "<<newest<<endl;
    
    
  }
}
/*
//******************************************************************** 
// METHOD NAME: Dequeue
// METHOD PURPOSE: Remove data from queue
// INPUT PARAMETERS: 
// data - of struct STUDENT to where data will be placed into from the queue
//RETURN VALUE � success
//********************************************************************
bool Queue::Dequeue(STUDENT &data)
{
bool status;
if(IsEmpty())
{
//cout<<"ERROR: Queue empty"<<endl;
status = false;
}
else
{
front = (front + 1) % queueSize;
data = queueArray[front];
n--;
status = true;
}
return status;
}
*/

//******************************************************************** 
// METHOD NAME: IsEmpty
// METHOD PURPOSE: Check if queue is empty
// INPUT PARAMETERS: 
// No input paramters
//RETURN VALUE � success
//********************************************************************
bool Queue::IsEmpty() const
{
bool status;
if(n != 0)
status = false;
else
status = true;
return status;
}


//******************************************************************** 
// METHOD NAME: IsFull
// METHOD PURPOSE: Check if queue is full
// INPUT PARAMETERS: 
// No input paramters
//RETURN VALUE � success
//********************************************************************
bool Queue::IsFull() const
{
  bool status;
  if(n < queueSize)
    status = false;
  else
    status = true;
  return status;
}

void Queue::get_Frame(int index, FRAME &frame){
  //if(IsEmpty())
    //  return 0;
  // else
  
  int j = newest;
  while(index>0){
     j = (j - 1);
     if(j==-1)
          j = queueSize-1;
     index--;
     }
     //imshow("GettingFrame", queueArray[j].image);
    frame = queueArray[j];
    
    return;
}

void Queue::displayQueue(){
     int i = 9;
     while(i>=0){
     cout<<"Displaying frame from slot "<<i<<endl;
     imshow("Frame", queueArray[i].image);
     waitKey(0);
     i--;
     }
     return;

}
