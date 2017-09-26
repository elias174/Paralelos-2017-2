#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv) {

    int numeroProcesadores,
            idProceso;
    long **A,
            *x,
            *y,
            *miFila;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numeroProcesadores);
    MPI_Comm_rank(MPI_COMM_WORLD, &idProceso);

    A = new long *[numeroProcesadores];
    x = new long [numeroProcesadores];
    if (idProceso == 0) {
        A[0] = new long [numeroProcesadores * numeroProcesadores];
        for (unsigned int i = 1; i < numeroProcesadores; i++) {
            A[i] = A[i - 1] + numeroProcesadores;
        }
        y = new long [numeroProcesadores];

        srand(time(0));
        for (unsigned int i = 0; i < numeroProcesadores; i++) {
            for (unsigned int j = 0; j < numeroProcesadores; j++) {
                if (j == 0) cout << "[";
                A[i][j] = rand() % 1000;
                cout << A[i][j];
                if (j == numeroProcesadores - 1) cout << "]";
                else cout << "  ";
            }
            x[i] = rand() % 100;
            cout << "\t  [" << x[i] << "]" << endl;
        }
        cout << "\n";
    }

    miFila = new long [numeroProcesadores];

    MPI_Scatter(A[0],
            numeroProcesadores,
            MPI_LONG,
            miFila,
            numeroProcesadores,
            MPI_LONG,
            0,
            MPI_COMM_WORLD);
    // Compartimos el vector entre todas los procesos
    MPI_Bcast(x,
            numeroProcesadores,
            MPI_LONG,
            0,
            MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    long subFinal = 0;
    for (unsigned int i = 0; i < numeroProcesadores; i++) {
        subFinal += miFila[i] * x[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&subFinal,
            1,
            MPI_LONG,
            y,
            1,
            MPI_LONG,
            0,
            MPI_COMM_WORLD);
    MPI_Finalize();

    if (idProceso == 0) {
        for (unsigned int i = 0; i < numeroProcesadores; i++) {
            cout << "\t" << y[i] << "\t" << endl;
        }
        delete [] y;
        delete [] A[0];
    }
    delete [] x;
    delete [] A;
    delete [] miFila;
}
