#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

int main(int argc, char** argv)
{

    int n=atoi(argv[2]);
    int thread_count=atoi(argv[1]);
    int *a = (int *)malloc(n * sizeof(int));
    for(int b=0; b<n; b++){
        srand(b);
        a[b] = rand() % 999;
    }
    int i, tmp,phase;
    double start = omp_get_wtime();
    for(phase=0; phase<n; phase++){
        if(phase%2==0)
#pragma omp parallel for num_threads(thread_count) default(none) shared(a,n) private(i,tmp)
            for(i=1; i<n; i+=2){
                if(a[i-1] > a[i]){
                    tmp = a[i-1];
                    a[i-1] = a[i];
                    a[i] = tmp;
                }
            }
        else
#pragma omp parallel for num_threads(thread_count) default(none) shared(a,n) private(i,tmp)
            for(i=1;i<n-1;i+=2){
                if(a[i] > a[i+1]){
                    tmp = a[i+1];
                    a[i+1] = a[i];
                    a[i] = tmp;
                }
            }
    }
    for(int b=0; b<n; b++){
        printf("%d ", a[b]);
    }
    printf("\n");
    printf("Time: \t %f \n", omp_get_wtime()-start);
    return 0;
}
