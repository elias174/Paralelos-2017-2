#include <math.h>
#include <iostream>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>
using namespace std;


#define TILE_WIDTH 4
#define N 10



// 4.7
__global__
void matMultKernel_tile_seven(int *d_M, int *d_N, int *d_P, int Width){

    extern __shared__ int Mds[];
    extern __shared__ int Nds[];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by*TILE_WIDTH + ty;
    int Col = bx*TILE_WIDTH + tx;

    float Pvalue = 0;
    int  ph,k;
    for(ph = 0; ph < Width/TILE_WIDTH; ++ph){
        // Collaborative loading of M and N tiles into shared memory
        if ( (Row<Width) && (ph*TILE_WIDTH+tx)<Width )
            Mds[ty][tx] = d_M[Row*Width + ph*TILE_WIDTH + tx];
        if ( (ph*TILE_WIDTH+ty)<Width && Col<Width )
            Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty) + Col];

        __syncthreads();
        for(k = 0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if ( (Row<Width) && (Col<Width))
        d_P[Row*Width + Col] = Pvalue;
}


// 4.6
__global__
void matMultKernel_tile_six(int *d_M, int *d_N, int *d_P, int Width){

    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by*TILE_WIDTH + ty;
    int Col = bx*TILE_WIDTH + tx;

    float Pvalue = 0;
    int  ph,k;
    for(ph = 0; ph < Width/TILE_WIDTH; ++ph){
        // Collaborative loading of M and N tiles into shared memory
        if ( (Row<Width) && (ph*TILE_WIDTH+tx)<Width )
            Mds[ty][tx] = d_M[Row*Width + ph*TILE_WIDTH + tx];
        if ( (ph*TILE_WIDTH+ty)<Width && Col<Width )
            Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty) + Col];

        __syncthreads();
        for(k = 0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if ( (Row<Width) && (Col<Width))
        d_P[Row*Width + Col] = Pvalue;
}



// 4.4
__global__
void MatrixMulKernel(float* d_M, float* d_N, float* d_P,
                     int Width) {
    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the d_P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    int Pvalue = 0;
    // Loop over the d_M and d_N tiles required to compute d_P element
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {

        // Collaborative loading of d_M and d_N tiles into shared memory
        Mds[ty][tx] = d_M[Row*Width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty)*Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {

            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    d_P[Row*Width + Col] = Pvalue;
}

//4.2
__global__ void matMultKernel(int *d_M, int *d_N, int *d_P, int Width){
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
    int Col = blockIdx.x*blockDim.x + threadIdx.x;
    int k = 0;
    if(Row < Width && Col < Width){
        float Pvalue = 0;
        for(k = 0; k < Width; ++k){
            Pvalue += d_M[Row*Width + k] * d_N[k*Width+Col];
        }
        d_P[Row*Width+Col] = Pvalue;
    }
}

void print_matrix(int matrix[N][N]){
    for(int i=0;i<N;i++){
        for(int j=0; j<N; j++){
            std::cout << matrix[i][j] << '\t';
        }
        std::cout << std::endl;
    }
}


int main()
{
    int a[N][N], b[N][N], c[N][N];
    int *dev_a, *dev_b, *dev_c;

    int size = N * N * sizeof(int);
    srand(time(NULL));
    for(int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            a[i][j] = rand() % 9;
            b[i][j] = rand() % 9;
        }
    }
    print_matrix(a);
    std::cout << std::endl;
    print_matrix(b);


    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(N/4.0),ceil(N/4.0),1);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
    matMultKernel_tile_six<<<dimGrid,dimBlock>>>(dev_a,dev_b,dev_c, N);
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    print_matrix(c);


    return 0;
}
