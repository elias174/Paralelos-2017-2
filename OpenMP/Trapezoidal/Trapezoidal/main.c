#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int thread_count;

double Trap(double a, double b, int n);
double f(double x);

int main(int argc, char* argv[]) {
   double  integral;
   double  a, b;
   int     n;

   thread_count = strtol(argv[1], NULL, 10);

   printf("Enter a, b, and n\n");
   scanf("%lf %lf %d", &a, &b, &n);

   integral = Trap(a, b, n);

   printf("With n = %d trapezoids, our estimate\n", n);
   printf("of the integral from %f to %f = %19.15e\n",
      a, b, integral);

   return 0;
}

double Trap(double a, double b, int n) {
   double  h, x, integral = 0.0;
   int  i;

   h = (b-a)/n;
   integral += (f(a) + f(b))/2.0;
#  pragma omp parallel for schedule(static) default(none) \
      shared(a, h, n) private(i, x) \
      reduction(+: integral) num_threads(thread_count)
   for (i = 1; i <= n-1; i++) {
      x = a + i*h;
      integral += f(x);
   }

   integral = integral*h;

   return integral;
}

double f(double x) {
   double return_val;

   return_val = x*x;
   return return_val;
}
