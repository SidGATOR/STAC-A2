#include "util.h"

int maxUpToNow( double *stock, double strike, unsigned paths, unsigned assets, 
   unsigned timesteps, double *retVal)
{
   double *assetMax = (double *) malloc( timesteps*sizeof(double));
   if( assetMax == NULL)
   {
      printf("Allocation failure\n");
      return -1;
   }
   for( int i=0; i<paths; i++)
   {
      for( int j=0; j<timesteps; j++)
      {
         assetMax[j] = 0.0;
         for( int k=0; k<assets; k++)
         {
            assetMax[j] = std::max( assetMax[j], stock[i*assets*timesteps 
               + k*timesteps + j]);
         }
         if( j>0){
            assetMax[j] = std::max( assetMax[j-1],assetMax[j]);
         }
         retVal[j*paths+i] = std::max( assetMax[j]- strike, 0.0);
      }
   }
   free(assetMax);
   return 0;
}

int removeElements( double *stock, double *inTheMoney, unsigned paths, 
   unsigned *retCount, double *retVal)
{
   int index=0;
   for( int i=0; i<paths; i++)
   {
      if( inTheMoney[i]!=0.0)
      {
         retVal[index++]=stock[i];
      }
   }   
   *retCount = index;
   return 0;
}

int restoreElements( double *stop, double *inTheMoney, unsigned paths, 
   double *retVal)
{
   int index=0;
   for( int i=0; i<paths; i++)
   {
      if( inTheMoney[i]==0.0)
      {
         retVal[i]=0.0;
      } else{
         retVal[i]=stop[index++];
      }
   }
   return 0;
}

int matrixMult( double *A, double *B, unsigned rowA, unsigned colA, 
   unsigned colB, double *C)
{
   double vector;
   for( int i=0; i<rowA; i++)
   {
      for( int j=0; j<colB; j++)
      {
         vector = 0.0;
         for( int k=0; k<colA; k++)
         {
            vector += A[i*colA + k]*B[k*colB + j];
         }
         C[i*colB + j] = vector;
      }
   }
}

int matrixMultA_BT( double *A, double *B, unsigned rowA, unsigned colA, 
   unsigned rowB, double *retVal)
{
   double vector;
   for( int i=0; i<rowA; i++)
   {
      for( int j=0; j<rowB; j++)
      {
         vector = 0.0;
         for( int k=0; k<colA; k++)
         {
            vector += A[i*colA + k]*B[j*colA + k];
         }
         retVal[i*rowB + j] = vector;
      }
   }
   return 0;
}

int polyVal( double *X, double *coef, unsigned numCoef, unsigned length, 
   double *retVal)
{
   for( int i=0; i<length; i++)
   {
      retVal[i] = 0.0;
      for( int j=numCoef-1; j>0; j--)
      {
         retVal[i] += pow(X[i],j)*coef[j];        
      }
      retVal[i] += coef[0];
   }
   return 0;
}

int mergePayoff( double *current, double *future, double *polyVal, 
   double *inTheMoney, unsigned paths, double r, double delta, double *retVal)
{
   double discount = exp(-r*delta);
   for( int i=0; i<paths; i++)
   {
      if( inTheMoney[i]!=0.0)
      {
         if( current[i]>polyVal[i])
         {
            retVal[i] = current[i];
         }else
         {
            retVal[i] = future[i]*discount;
         }
      }else{
         retVal[i] = future[i]*discount;
      }   
   }
   return 0;
}

int readFile( const char *filename, unsigned size, double *retVal)
{
   FILE *fp = fopen(filename,"r");
   if( fp == NULL)
   {
      printf("File I/O error\n");
      return -1;
   }
   char value[100];
   for( int i=0; i<size; i++)
   {
      fscanf(fp,"%s",&value);
      retVal[i] = atof(value);
   }
   return 0;
}

int pinv( double *diag, unsigned size, double *retVal)
{
   unsigned index=0;
   for( int i=0; i<size*size; i++)
   {
      if( i == size*index+index)
      {
         retVal[i] = 1.0/diag[index++];
      }else{
         retVal[i] = 0.0;
      }
   }
   return 0;
}

