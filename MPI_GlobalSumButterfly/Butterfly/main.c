#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

const int MAX_CONTRIB = 10;

int Global_sum(int my_contrib, int my_rank, int p, MPI_Comm comm);

int main(int argc, char* argv[]) {
   int      p, my_rank;
   MPI_Comm comm;
   int      my_contrib;
   int      sum;

   MPI_Init(&argc, &argv);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &p);
   MPI_Comm_rank(comm, &my_rank);

   srandom(my_rank+1);
   my_contrib = random() % MAX_CONTRIB;
   printf("Proc %d > my_contrib = %d\n", my_rank, my_contrib);
   sum = Global_sum(my_contrib, my_rank, p, comm);
   if (my_rank == 0)
      printf("Proc %d > global sum = %d\n", my_rank, sum);

   MPI_Finalize();
   return 0;
}

int Global_sum(int my_contrib, int my_rank, int p, MPI_Comm comm) {
    int        sum = my_contrib;
    int        temp;
    int        partner;
    unsigned   bitmask = 1;

    while (bitmask < p) {
        partner = my_rank ^ bitmask;
        MPI_Sendrecv(&sum, 1, MPI_INT, partner, 0,
                    &temp, 1, MPI_INT, partner, 0,
                    comm, MPI_STATUS_IGNORE);
        sum += temp;
        bitmask <<= 1;
    }

    return sum;
}
