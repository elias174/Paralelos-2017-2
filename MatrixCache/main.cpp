 #include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>

using namespace std;

const int MAX = 600;

#define MIN(a, b) (((a) < (b)) ? (a) : (b))


void print_matrix(double matrix[][MAX]){
    for(int i=0; i<MAX; i++){
        for(int j=0; j<MAX; j++){
            cout << matrix[i][j] << '\t';
        }
        cout << endl;
    }
}

void print_array(double array[]){
    for(int i=0; i<MAX; i++){
        cout << array[i] << '\t';
    }
}

void cache_matrix_acces_1(){
    double A[MAX][MAX], x[MAX], y[MAX];
    srand (time(NULL));
    for(int i=0; i<MAX; i++){
        for(int j=0; j<MAX; j++){
            A[i][j] = rand() % 1000 + 1;
        }
        x[i] = rand() % 1000 + 1;
        y[i] = 0;
    }

    // First loop
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<MAX; i++){
        for(int j=0; j<MAX; j++){
            y[i] = A[i][j]*x[j];
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "First Loop:" << elapsed.count() << " s\n";

    cout << endl << endl;

    std::fill(y,y+MAX, 0);

    auto start_1 = std::chrono::high_resolution_clock::now();
    for(int j=0; j<MAX; j++){
        for(int i=0; i<MAX; i++){
            y[i] = A[j][i]*x[j];
        }
    }
    auto finish_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_1 = finish_1 - start_1;
    std::cout << "Second Loop: " << elapsed_1.count() << " s\n";

}

void matrix_multiplication_3loops(){
    double A[MAX][MAX], B[MAX][MAX], C[MAX][MAX], D[MAX][MAX];
    srand (time(NULL));
    for(int i=0; i<MAX; i++){
        for(int j=0; j<MAX; j++){
            A[i][j] = rand() % 1000 + 1;
            B[i][j] = rand() % 1000 + 1;
        }
    }

    // classic 3 loop
    auto start_1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<MAX; i++){
        for(int j=0; j<MAX; j++){
            C[i][j] = 0;
            for(int k=0; k<MAX; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    }
    auto finish_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_1 = finish_1 - start_1;
    std::cout << "3 loop matrix multiplication " << elapsed_1.count() << " s\n";

    int blockSize=2;
    int n = MAX;
    //six loop
    auto start = std::chrono::high_resolution_clock::now();
    for (int ii = 0; ii<n; ii+=blockSize) {
        for (int jj = 0; jj<n; jj+=blockSize) {
            for (int kk = 0; kk<n; kk+=blockSize) {
                for (int i = ii; i<MIN(ii+blockSize,n); i++) {
                    for (int j = jj; j<MIN(jj+blockSize, n); j++) {
                        for (int k = kk; k<MIN(kk+blockSize, n); k++) {
                            D[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "6 loop matrix multiplication " << elapsed.count() << " s\n";

}

int main(int argc, char *argv[])
{
    cache_matrix_acces_1();
    cout << "Hello World!" << endl;
    return 0;
}
