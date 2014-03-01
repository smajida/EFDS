#ifndef QUEUE_H
#define QUEUE_H
#include "Frame.h"
#include <iostream>
#include <ostream>
#include <iomanip>

using namespace std;

class Queue
{
     private:
         FRAME *queueArray;  //struct FRAME holding data about images
         int queueSize;        //size of the queue
         int oldest;            //front of the queue
         int newest;             //back of the queue
         int n;                //number of items in the queue

     public:
         Queue(int);
         Queue(const Queue &);
         ~Queue();
         void Enqueue (FRAME);
         //bool Dequeue (FRAME &); //Do not need
         bool IsEmpty() const; //Do not need
         bool IsFull() const;
	  void get_Frame(int, FRAME&);
	  void displayQueue();
};



#endif /* QUEUE_H_ */
