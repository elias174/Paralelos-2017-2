#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define NUM_THREADS 4         //number of threads
#define ITERATIONS 100000000
#define N 1E8
#define d 1E-8


int sum;
pthread_mutex_t count_mutex;

void *calculate(void *thread_id){
    double x,y;
    int result=0;
    int rank = (int)thread_id;
    float tot_iterations = ITERATIONS/NUM_THREADS;
    srand((int)time(0));
    for (int i=0; i<tot_iterations; i+=1)
    {
        x=rand()/(RAND_MAX+1.0);
        y=rand()/(RAND_MAX+1.0);
        if(x*x+y*y<1.0){
            result += 1;
        }
    }
    pthread_mutex_lock(&count_mutex);
    sum += result;
    pthread_mutex_unlock(&count_mutex);

}


int main(int argc, char *argv[])
{
    pthread_t threads[NUM_THREADS];
    int rc;
    long t;
    void *b = NULL;
    double pi;
    for(t=0;t<NUM_THREADS;t++){
      rc = pthread_create(&threads[t], NULL, calculate, (void *)t);
    }
    for(t=0;t<NUM_THREADS;t++){
        pthread_join(threads[t], &b);
    }
    pi = 4*(double)sum/ITERATIONS;
    //pi = sum/ITERATIONS*4;
    printf("Result: %g\n", pi);
    return 0;
}
