#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>

using namespace std;

void print_matrix(int **matrix, int m, int n){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            cout << matrix[i][j] << "\t";
        }
        cout << endl;
    }
}

void print_array(int *array, int n){
    for(int i=0; i<n; i++){
        cout << array[i] << '\t';
    }
    cout << endl;
}

void initialize_matrix(int **matrix, int m, int n){
    for(int i=0; i<m; i++){
        matrix[i] = new int[n];
        for(int j=0; j<n; j++){
            matrix[i][j] = rand() % 999;
        }
    }
}

void initialize_array(int *array, int n){
    for(int i=0; i<n; i++){
        array[i] = rand()%999;
    }
}

int main(int argc, char** argv)
{
    srand((int)time(0));
    int thread_count = atoi(argv[3]);
    int m=atoi(argv[1]);
    int n=atoi(argv[2]);

    int **matrix = new int*[m];
    int *vector = new int[n];
    int *result = new int[n];
    initialize_matrix(matrix, m, n);
    initialize_array(vector, n);
    int i,j;
    double start = omp_get_wtime();
#pragma omp parallel for num_threads(thread_count) default(none) private(i, j) shared(matrix, vector, result, m, n)
    for(i=0; i<m; i++){
        result[i] = 0;
        for(j=0; j<n; j++)
            result[i] += matrix[i][j]*vector[j];
    }
    printf("Time: \t %f \n", omp_get_wtime()-start);

    return 0;
}
