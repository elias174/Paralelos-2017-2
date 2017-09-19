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
    int divisor = 2;
    int core_diff = 1;
    int partner;
    int received_value;

    while (divisor <= p) {
        if(my_rank % divisor == 0){
            partner = my_rank + core_diff;
            MPI_Recv(&received_value, 1, MPI_INT, partner, 0, comm, MPI_STATUS_IGNORE);
            sum += received_value;
        }
        else if ((my_rank % (divisor/2)) == 0){
            partner = my_rank - core_diff;
            MPI_Send(&sum, 1, MPI_INT, partner, 0, comm);
        }
        divisor *= 2;
        core_diff *= 2;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return sum;
}
