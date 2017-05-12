#include <stdlib.h>
#include <stdio.h>
#include "option.h"
#include "main.h"

int simpleCall(int assets, int timesteps, int paths, void *input, void *K, void *output, int *retSize)
{
	output	= (OPTDTYPE*) malloc(assets*paths*sizeof(OPTDTYPE));
	for(int i=0; i<paths; i++)
	{
		for(int j=0; j<assets; j++)
		{
			((OPTDTYPE*)output)[i*assets+j] = ((OPTDTYPE*)input)[i*assets+j] - *(OPTDTYPE*)K;
		}
	}
	*retSize = assets;
	return 0;
}
int eurPayoff(int assets, int timesteps, int paths, double r, double T, void *input, void *K, void *output, int *retSize, 
	int(*payoff)(int,int,int,void*,void*,void*,int*))
{

	OPTDTYPE *timeT			= (OPTDTYPE*) malloc(assets*paths*sizeof(OPTDTYPE));
	output					= (OPTDTYPE*) calloc(assets,sizeof(OPTDTYPE));
	OPTDTYPE *exerciseVal;
	unsigned maturity		= (unsigned)(T*(double)TIMESTEPS);
	int tempSize;

	for(int i=0; i<paths; i++)
	{
		for(int j=0; j<assets; j++)
		{
			timeT[i*assets+j] = ((OPTDTYPE*)input)[i*assets*timesteps+j*timesteps+maturity];
		}
	}

	(*payoff)(assets,timesteps,paths,(void*)timeT,K,(void*)exerciseVal,&tempSize);


	free(timeT);
	free(exerciseVal);
	return 0;
}
int amerPayoff(int assets, int timesteps, int paths, double r, double T, void *input, void *K, void *output, int *retSize, 
	int(*payoff)(int,int,int,void*,void*,void*,int*))
{
	return 0;
}
