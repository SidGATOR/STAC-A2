#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "util.h"
#define NUMCOEF 3

extern int svd(double (*a)[3], int m, int n, double *w, double (*v)[3]);

int main (int argc, char **argv)
//int main(void)
{

  if(argc != 3)
  {
	printf("ERROR: INCORRECT NUMBER OF  <ARGS>\n");
	printf("ARGUMENT FORMAT: <NUMBER OF PATHS> <NUMBER OF FPGAS> \n");
	return -1;
  }
  
  //unsigned paths = 25000;
   unsigned paths = atoi(argv[1]); 
   unsigned timesteps = 252;
   unsigned assets = 5;

   double strike = 1.0;
   double r = 0.02;
   double delta = 1/timesteps;

   struct timeval st;
   // For SVD calculation
   double *w = (double*) malloc( NUMCOEF*sizeof(double));
   double (*a)[NUMCOEF] = (double(*)[NUMCOEF]) malloc( paths*sizeof(*a));
   double (*v)[NUMCOEF] = (double(*)[NUMCOEF]) malloc( NUMCOEF*sizeof(*v));
   // Stock prices from simulation
   double *inputVal = (double*) malloc(paths*timesteps*assets*sizeof(double));
   // For stock payoff calculation (retruns pathXtimesteps
   double *payoff = (double*) malloc(paths*timesteps*sizeof(double));
   // Used in each timestep for optimal payoff strategy
   double *inTheMoney = (double*) malloc(paths*sizeof(double));
   double *bias = (double*) malloc(paths*sizeof(double));
   double *futurePayoff = (double*) malloc(paths*sizeof(double));
   double *X = (double*) malloc(paths*sizeof(double));
   double *polyX = (double*) malloc(paths*sizeof(double));
   double *futureVal = (double*) malloc(paths*sizeof(double));
   // Risk-free discount factor
   double discount = exp(-r*delta);
   if(atoi(argv[2]) == 1)
   {
   printf("Calculating the option price via 1 FPGA\n");
   readFile("../newTest/test/PATHGENout1.csv", paths*timesteps*assets, inputVal);
   }
   else
   {
   if(atoi(argv[2]) == 2 )
   {
   printf("Calculating the option price via 2 FPGAs\n");
   readFile("../newTest/test/PATHGENoutP3.csv", paths*timesteps*assets, inputVal);
   }
   else
   {
   printf("Running default case of 1 FPGA\n");
   readFile("../newTest/test/PATHGENout.csv", paths*timesteps*assets, inputVal);
   }
   }
   // Calcuate stock payoff
   maxUpToNow( inputVal, strike, paths, assets, timesteps, payoff);

   unsigned count;
   double winv[NUMCOEF*NUMCOEF];
   double v_invW[NUMCOEF*NUMCOEF];
   double *reg0 = (double*) malloc(paths*NUMCOEF*sizeof(double));
   double retCoef[NUMCOEF];
  
   for( int i=0; i<paths; i++)
   {
      futureVal[i] = payoff[(timesteps-1)*paths + i];
   }
 
for( int l=timesteps-2; l>=0; l--)
{   

   for( int i=0; i<paths; i++)
   {
      futurePayoff[i] = futureVal[i]*discount;
   }

   for( int i=0; i<paths; i++)
   {
      if( payoff[l*paths + i]>0.0)
      {
         inTheMoney[i] = payoff[l*paths + i];
      } else{
         inTheMoney[i] = 0.0;
      }
   }

   removeElements( &payoff[l*paths], inTheMoney, paths, &count, bias);
   
   for( int i=0; i<count; i++)
   {
      a[i][0] = 1.0;
      for( int j=1; j<NUMCOEF; j++)
      {
         a[i][j] = pow( bias[i] + strike, j);
      }
   } 

   svd( a, (int) count, NUMCOEF, w, v);
  
   pinv( w, NUMCOEF, winv);

   matrixMult( (double*) v, winv, NUMCOEF, NUMCOEF, NUMCOEF, v_invW);

   matrixMultA_BT( v_invW, (double*) a, NUMCOEF, NUMCOEF, paths, reg0);

   matrixMult( reg0, futurePayoff, NUMCOEF, paths, 1, retCoef);

   for( int i=0; i<paths; i++)
   {
      X[i] = payoff[l*paths + i]+strike;
   }

   polyVal( X, retCoef, NUMCOEF, paths, polyX);

   mergePayoff( &payoff[l*paths], futurePayoff, polyX, inTheMoney, paths, 
      r, delta, futureVal);

  /*
   printf("\n\nFuture Value %i\n",l);

   for( int i=0; i<paths; i++)
   {
      printf("%f\n",futureVal[i]);
   }
   */
}
   double optionPrice = 0.0;
   for( int i=0; i<paths; i++)
   {
      optionPrice += futureVal[i];
   }
   optionPrice /= (double) paths;
   optionPrice *= discount; 

   printf("Option Price: %f\n", optionPrice);

   free(w);
   free(v);
   free(a);
   free(inputVal);
   free(payoff);
   free(inTheMoney);
   free(bias);
   free(reg0);
   free(futurePayoff);
   free(X);
   free(polyX);
   free(futureVal);
   return 0;
}

