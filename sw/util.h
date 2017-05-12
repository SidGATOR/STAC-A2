#ifndef UTIL_INCLUDED
#define UTIL_INCLUDED
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

int maxUpToNow( double *stock, double strike, unsigned paths, unsigned assets, 
   unsigned timesteps, double *retVal);

int removeElements( double *stock, double *inTheMoney, unsigned paths, 
   unsigned *retCount, double *retVal);

int restoreElements( double *stop, double *inTheMoney, unsigned paths, 
   double *retVal);

int matrixMult( double *A, double *B, unsigned rowA, unsigned colA, 
   unsigned colB, double *C);

int matrixMultA_BT( double *A, double *B, unsigned rowA, unsigned colA, 
   unsigned rowB, double *retVal);

int polyVal( double *X, double *coef, unsigned numCoef, unsigned length, 
   double *retVal);

int mergePayoff( double *current, double *future, double *polyVal, 
   double *inTheMoney, unsigned paths, double r, double delta, double *retVal);

int readFile( const char *filename, unsigned size, double *retVal);

int pinv( double *diag, unsigned size, double *retVal);
 
#endif

