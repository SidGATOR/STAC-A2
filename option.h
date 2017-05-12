typedef double OPTDTYPE;
int simpleCall(int assets, int timesteps, int paths, double r, double T, void *input, void *K, void *output, int *retSize);
int eurPayoff(int assets, int timesteps, int paths, double r, double T,void *input, void *K, void *output, int *retSize, 
	int(*payoff)(int,int,int,void*,void*,void*,int*));
int amerPayoff(int assets, int timesteps, int paths, void *input, void *K, void *output, int *retSize, 
	int(*payoff)(int,int,int,void*,void*,void*,int*));