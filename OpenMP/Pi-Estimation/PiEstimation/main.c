#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char** argv)
{
    int thread_count=atoi(argv[1]);
    double sum = 0.0;
    double factor;
    int k;
    int n=10000000;
#pragma omp parallel for num_threads(thread_count) reduction(+:sum) private(factor)
    for(k=0;k<n;k++){
        if(k%2 == 0)
            factor = 1.0;
        else
            factor = -1.0;
        sum += factor/(2*k+1);
    }
    double pi = 4.0*sum;
    printf("%f \n", pi);
    return 0;
}
