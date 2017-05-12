/*
Kenneth Hill
STAC-A2 main 
*/
#include "model.h"
#include "option.h"
#include "openclBuilder.h"
#include "openclBuffer.h"


typedef float hostD;
typedef cl_float kernelD;

#define EPSILON (1e-4f)

void cleanup(void);
void CL_CALLBACK openclError(const char *errinfo, const void *private_info, size_t cb, void *user_data);
size_t nextPow2(size_t num);

int main( int argc, char **argv  ) {

  cl_uint num_platforms;
  cl_uint num_devices;
   
  #ifndef MODEL_DEBUG
	// TODO I/O from input file
  #endif
  
  input2 = acl_aligned_malloc( sizeof(kernelD)*4*ASSETS*ASSETS);
  if( input2 == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  // correlation matrix
  memcpy(input2, input_A, 4*ASSETS*ASSETS*sizeof(kernelD));

  kappa = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( kappa == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  invKappa = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( invKappa == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  theta = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( theta == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  xi = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( xi == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  xi2 = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( xi2 == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  expKD = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( expKD == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  nexpKD = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( nexpKD == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  nexpKD2 = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( nexpKD2 == NULL)
  {
	  printf("Host allcoation failure\n");
	  return -1;
  }
  S0 = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( S0 == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  V0 = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( V0 == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  rho = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( rho == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  invsqrtRho = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( invsqrtRho == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  K0 = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( K0 == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  K1 = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( K1 == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  K2 = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( K2 == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  K3 = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( K3 == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }
  K4 = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( K4 == NULL)
  {
	  printf("Host allocation failure\n");
	  return -1;
  }

  // set up Heston model 
  for( int i=0; i<ASSETS; i++)
  {
	((hostD *)kappa)[i] 		= input_kappa[i];
	((hostD *) invKappa)[i]		= 1.0/input_kappa[i];
	((hostD *) theta)[i]		= input_theta[i];
	((hostD *) xi)[i]			= input_xi[i];
	((hostD *) xi2)[i]			= pow(input_xi[i],2);
	((hostD *) expKD)[i]		= exp( -1.0*((hostD *) kappa)[i]*delta);
	((hostD *) nexpKD)[i]		= ( 1.0 - ((hostD *) expKD)[i]);
	((hostD *) nexpKD2)[i]		= pow( ((hostD *) nexpKD)[i], 2);
	((hostD *) S0)[i]			= input_Y0[i];
	((hostD *) V0)[i]			= input_V0[i];
	((hostD *) rho)[i]			= input_rho[i];
	((hostD *) invsqrtRho)[i]	= 1.0 / sqrt(1.0 - pow( ((hostD *) rho)[i], 2));
	((hostD *) K0)[i]			= -(delta*((hostD *) theta)[i]*((hostD *) kappa)[i]*((hostD *) rho)[i])/((hostD *) xi)[i];
	((hostD *) K1)[i]			= gamma1*delta*( (((hostD *) kappa)[i]*((hostD *) rho)[i]/((hostD *) xi)[i]) - 0.5) 
		- (((hostD *) rho)[i]/((hostD *) xi)[i]);
	((hostD *) K2)[i]			= gamma2*delta*( (((hostD *) kappa)[i]*((hostD *) rho)[i]/((hostD *) xi)[i]) - 0.5) 
		+ (((hostD *) rho)[i]/((hostD *) xi)[i]);
	((hostD *) K3)[i]			= gamma1*delta*(1.0 - pow(((hostD *) rho)[i], 2));
	((hostD *) K4)[i]			= gamma2*delta*(1.0 - pow(((hostD *) rho)[i], 2));
	
  }


  
  // get the openCL platform information
  status = clGetPlatformIDs(1, &platform, &num_platforms);
  if(status != CL_SUCCESS) {
    dump_error("Failed clGetPlatformIDs.", status);
    cleanup();
    return -1;
  }
  if(num_platforms != 1) {
    printf("Found %d platforms!\n", num_platforms);
    cleanup();
    return -1;
  }


  //Number of Devices//

  // get the openCL device ID
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, &num_devices);
  if(status != CL_SUCCESS) {
    dump_error("Failed clGetDeviceIDs.", status);
    cleanup();
    return -1;
  }
  /*
#ifdef ALTERA_CL
// Altera currently supports one device per server
  if(num_devices != 1) {
    printf("Found %d devices!\n", num_devices);
    cleanup();
    return -1;
  }
#endif
*/
  cl_ulong memAllocSize;
  status = clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
	  &memAllocSize,NULL);
  if(status != CL_SUCCESS)
  {
	  dump_error("Failed max alloc size",status);
	  return -1;
  }
  printf("Max alloc size: %d MB\n", memAllocSize/1024/1024);

  cl_ulong memGlobalSize;
  status = clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
	  &memGlobalSize,NULL);
  if(status != CL_SUCCESS)
  {
	  dump_error("Failed max alloc size",status);
	  return -1;
  }
  printf("Global memory size: %d MB\n", memGlobalSize/1024/1024);

  // create a context
  context = clCreateContext(0, 1, &device, &openclError, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateContext.", status);
    cleanup();
    return -1;
  }

  // create command queues for each operation
  queuePATHGEN = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateCommandQueue.", status);
    cleanup();
    return -1;
  }

#ifndef ALTERA_CL  
  // create OpenCL programs

  if( createProgram("alteraPATHGENnew.cl", &programPATHGEN, context, device) != 0)
  {
	printf("Failed to create programPATHGEN.");
	cleanup();
	return -1;
  }
  
#else

  if( createProgram("alteraPATHGENnew.aocx", &programPATHGEN, context, device) != 0)
  {
	printf("Failed to create programPATHGEN.");
	cleanup();
	return -1;
  }

#endif
  
  // OpenCL kernels
  
  // execution timers
  double start,end;
  // openCL workgroup information
  size_t gSize[3] = { paths, ASSETS, 1};
  size_t gRun[3] = { paths, ASSETS, 1};
  size_t lSize[3] = { 1, ASSETS, 1};
  // openCL workload adjustment
  size_t runN;
  // runtime buffers
  cl_mem *runKernelBuf	= (cl_mem*) malloc(21*sizeof(cl_mem));
  void **runBuffer		= (void**) malloc(21*sizeof(void*));
  // openCL memReference Count
  cl_uint refCount;
  cl_uint memFree;
  cl_uint flag;
#ifdef PLOT
  FILE *fp;
#endif

  // kernel timers
  double timeSQRT, timeEXP, timeLOG, timeUNR, timeCORRAND, timePATHGEN;

  // run PATHGEN
  // create pathgen kernel
  kernelPATHGEN = clCreateKernel(programPATHGEN, "alteraPATHGEN", &status);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clCreateKernel alteraPATHGENnew.", status);
	return -5;
  }
#ifdef PLOT
 // for( paths = 10000; paths < 220000 ;paths = paths + 7000)
  {
  	paths = 2000;
	//  paths = atoi(argv[1]);
  	N             = 2*assets*paths*(unsigned ( hostD (TIMESTEPS) *T));
	printf("For %d paths\n", paths);
#else
  {
   printf("For Path: %d\n",paths);
#endif
  // Configure FPGA
  gSize[0] = paths;
  gSize[1] = ASSETS;
  gSize[2] = 1;
  
  if( clGetKernelWorkGroupInfo( kernelPATHGEN, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, 3*sizeof(size_t), lSize, NULL) != CL_SUCCESS)
  {
	  printf("Failed to find cl workgroup size!\n");
	  cleanup();
	  return -1;
  }  
  
  openclWork( lSize, gSize, gRun, &runN);
  runN *= 2*totalTimesteps;

  start = timerval();
  //Random Seed//
  runKernelBuf[0]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(cl_uint), NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
 
  //Random numbers//
  runKernelBuf[1]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)* ICDFSize, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  
  //kappa//
  runKernelBuf[2]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //invKappa//
  runKernelBuf[3] = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer ICDF",status);
	return -1;
  }
  //theta//
  runKernelBuf[4]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //xi//
  runKernelBuf[5]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //xi2//
  runKernelBuf[6]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //expKD//
  runKernelBuf[7]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //nexpKD//
  runKernelBuf[8]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //nexpKD2//
  runKernelBuf[9]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //S0//
  runKernelBuf[10]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //V0//
  runKernelBuf[11]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //rho//
  runKernelBuf[12]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //invsqrtRho//
  runKernelBuf[13]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //K0//
  runKernelBuf[14]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //K1//
  runKernelBuf[15]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //K2//
  runKernelBuf[16]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //K3//
  runKernelBuf[17]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //K4//
  runKernelBuf[18]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }	
 //retVolPaths//
  runKernelBuf[19]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*(runN/2), NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
 //retPricePaths//
  runKernelBuf[20]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*(runN/2), NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  
  if(initPATHGEN(programPATHGEN, &kernelPATHGEN, runKernelBuf, 21, delta, totalTimesteps) != 0)
  {
	  printf("Failed initPATHGEN\n");
	  cleanup();
	  return -1;
  } 
  
  runBuffer[0]		= acl_aligned_malloc( sizeof(cl_uint));
  if( runBuffer[0] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[0],(void*) &seed, sizeof(cl_uint));
  
  runBuffer[1]		= acl_aligned_malloc( sizeof(kernelD)*ICDFSize);
  if( runBuffer[1] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  ICDFset( (float *) runBuffer[1], "testICDF.csv", (unsigned) ICDFSize);
  
  runBuffer[2]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[2] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[2], kappa, sizeof(kernelD)*ASSETS);
  
  runBuffer[3]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[3] == NULL)
  {
	printf("Host allcation failure\n");
	return -1;
  }
  memcpy( runBuffer[3], invKappa, sizeof(kernelD)*ASSETS);
  
  runBuffer[4]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[4] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  
  memcpy( runBuffer[4], theta, sizeof(kernelD)*ASSETS);
  
  runBuffer[5]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[5] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[5], xi, sizeof(kernelD)*ASSETS);
  
  runBuffer[6]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[6] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[6], xi2, sizeof(kernelD)*ASSETS);
  
  runBuffer[7]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[7] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[7], expKD, sizeof(kernelD)*ASSETS);
  
  runBuffer[8]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[8] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[8], nexpKD, sizeof(kernelD)*ASSETS);
  
  runBuffer[9]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[9] == NULL)
  {
    printf("Host allocation failure\n");
    return -1;	
  }
  memcpy( runBuffer[9], nexpKD2, sizeof(kernelD)*ASSETS);
  
  runBuffer[10]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[10] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[10], S0, sizeof(kernelD)*ASSETS);
  
  runBuffer[11]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[11] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[11], V0, sizeof(kernelD)*ASSETS);
  
  runBuffer[12]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[12] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[12], rho, sizeof(kernelD)*ASSETS);
  
  runBuffer[13]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[13] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[13], invsqrtRho, sizeof(kernelD)*ASSETS);
  
  runBuffer[14]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[14] == 	NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[14], K0, sizeof(kernelD)*ASSETS);
  
  runBuffer[15]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[15] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[15], K1, sizeof(kernelD)*ASSETS);
  
  runBuffer[16]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[16] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[16], K2, sizeof(kernelD)*ASSETS);
  
  runBuffer[17]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[17] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[17], K3, sizeof(kernelD)*ASSETS);
  
  runBuffer[18]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[18] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[18], K4, sizeof(kernelD)*ASSETS);
  
  runBuffer[19]		= acl_aligned_malloc( sizeof(kernelD)*(N/2));
  if( runBuffer[19] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  
  runBuffer[20]		= acl_aligned_malloc( sizeof(kernelD)*(N/2));
  if( runBuffer[20] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  
  if( runPATHGEN(gRun, lSize, N, queuePATHGEN, runKernelBuf, runBuffer, 21, &kernelPATHGEN, &kernel_event, &timePATHGEN) != 0)
  {
	printf("Failed running PATHGEN");
	cleanup();
	return -1;
  }

  end = timerval();
  //writeOutput( (double *) runBuffer[20], (N/2), "PATHGENout.csv");
  printf("PATHGEN execution WALL time: %f\nCL time: %f\n", end - start, execution_time(kernel_event) );
#ifdef PLOT
  fp = fopen("TimingKernel.txt","a");
  fprintf(fp,"%d\t%f\n",paths,execution_time(kernel_event));
  fclose(fp);
#endif
#ifndef NO_OUTPUT
  printf("Generating Output from FPGA\n");
  writeOutput( (float *) runBuffer[20], (N/2), "PATHGENout.csv");
#endif
  // operation cleanup
  //clReleaseKernel( kernelPATHGEN);  
  for( int i=0; i<21; i++)
  {
	clReleaseMemObject(runKernelBuf[i]);
	acl_aligned_free(runBuffer[i]);
  }
#ifdef ALTERA_CL
  flag = 0;
  memFree = 0;
  do
  {
    if( memFree == 20)
	{
		flag = 1;
	}
    //memFree = 0;
	for( int i=0; i<21; i++)
	{
		clReleaseMemObject(runKernelBuf[i]);
		clGetMemObjectInfo(runKernelBuf[i], CL_MEM_REFERENCE_COUNT, sizeof(cl_uint),
			&refCount, NULL);
		//memFree += refCount;
   memFree = i;
	}
  }while(memFree != 20 || flag != 1);
#endif
  } //This Bracket is for the for loop//
  clReleaseKernel( kernelPATHGEN);
  clReleaseEvent( kernel_event);
#ifdef ALTERA_CL
  // free the resources allocated
  cleanup();
#endif

  //char prompt;
 // clReleaseEvent(kernel_event);
  return 0;
}

// free the resources allocated during initialization
void cleanup(void) {
  if(kernelSQRT) 
    clReleaseKernel(kernelSQRT);
  if(kernelEXP)
    clReleaseKernel(kernelEXP);
  if(kernelLOG)
	clReleaseKernel(kernelLOG);
  if(kernelUNR)
	clReleaseKernel(kernelUNR);
  if(kernelCORRAND)
	clReleaseKernel(kernelCORRAND);
  if(kernelPATHGEN)
	clReleaseKernel(kernelPATHGEN);
  if(programSQRT) 
    clReleaseProgram(programSQRT);
  if(programEXP)
	clReleaseProgram(programEXP);
  if(programLOG)
	clReleaseProgram(programLOG);
  if(programUNR)
	clReleaseProgram(programUNR);
  if(programCORRAND)
	clReleaseProgram(programCORRAND);
  if(programPATHGEN)
	clReleaseProgram(programPATHGEN);
  if(queueSQRT) 
    clReleaseCommandQueue(queueSQRT);
  if(queueEXP) 
    clReleaseCommandQueue(queueEXP);
  if(queueLOG) 
    clReleaseCommandQueue(queueLOG);
  if(queueUNR) 
    clReleaseCommandQueue(queueUNR);
  if(queueCORRAND) 
    clReleaseCommandQueue(queueCORRAND);
  if(queuePATHGEN) 
    clReleaseCommandQueue(queuePATHGEN);
  if(context) 
    clReleaseContext(context);
  if(kernel_ICDF)
	clReleaseMemObject(kernel_ICDF);
  if(kernel_returnValues) 
    clReleaseMemObject(kernel_returnValues);
  if(kernel_input1)
	clReleaseMemObject(kernel_input1);
  if(kernel_input2)
	clReleaseMemObject(kernel_input2);
  if(kernel_rnArray)
	clReleaseMemObject(kernel_rnArray);
  if(kernel_unArray)
	clReleaseMemObject(kernel_unArray);
  if(kernel_kappa)
	clReleaseMemObject(kernel_kappa);
  if(kernel_invKappa)
	clReleaseMemObject(kernel_invKappa);
  if(kernel_theta)
	clReleaseMemObject(kernel_theta);
  if(kernel_xi)
	clReleaseMemObject(kernel_xi);
  if(kernel_xi2)
	clReleaseMemObject(kernel_xi2);
  if(kernel_expKD)
	clReleaseMemObject(kernel_expKD);
  if(kernel_nexpKD)
	clReleaseMemObject(kernel_nexpKD);
  if(kernel_nexpKD2)
	clReleaseMemObject(kernel_nexpKD2);
  if(kernel_S0)
	clReleaseMemObject(kernel_S0);
  if(kernel_V0)
	clReleaseMemObject(kernel_V0);
  if(kernel_rho)
	clReleaseMemObject(kernel_rho);
  if(kernel_invsqrtRho)
	clReleaseMemObject(kernel_invsqrtRho);
  if(kernel_K0)
	clReleaseMemObject(kernel_K0);
  if(kernel_K1)
	clReleaseMemObject(kernel_K1);
  if(kernel_K2)
	clReleaseMemObject(kernel_K2);
  if(kernel_K3)
	clReleaseMemObject(kernel_K3);
  if(kernel_K4)
	clReleaseMemObject(kernel_K4);
  if(kernel_retVolPaths)
	clReleaseMemObject(kernel_retVolPaths);
  if(kernel_retPricePaths)
	clReleaseMemObject(kernel_retPricePaths);
  if(returnValues) 
    acl_aligned_free(returnValues);
  if(ICDF)
	acl_aligned_free(ICDF);
  if(input1)
	acl_aligned_free(input1);
  if(input2)
	acl_aligned_free(input2);
  if(rnArray)
	acl_aligned_free(rnArray);
  if(unArray)
	acl_aligned_free(unArray);
  if(kappa)
	acl_aligned_free(kappa);
  if(invKappa)
	acl_aligned_free(invKappa);
  if(theta)
	acl_aligned_free(theta);
  if(xi)
	acl_aligned_free(xi);
  if(xi2)
	acl_aligned_free(xi2);
  if(expKD)
	acl_aligned_free(expKD);
  if(nexpKD)
	acl_aligned_free(nexpKD);
  if(nexpKD2)
	acl_aligned_free(nexpKD2);
  if(S0)
	acl_aligned_free(S0);
  if(V0)
	acl_aligned_free(V0);
  if(rho)
	acl_aligned_free(rho);
  if(invsqrtRho)
	acl_aligned_free(invsqrtRho);
  if(K0)
	acl_aligned_free(K0);
  if(K1)
	acl_aligned_free(K1);
  if(K2)
	acl_aligned_free(K2);
  if(K3)
	acl_aligned_free(K3);
  if(K4)
	acl_aligned_free(K4);
  if(retVolPaths)
	acl_aligned_free(retVolPaths);
  if(retPricePaths)
	acl_aligned_free(retPricePaths);
}

void CL_CALLBACK openclError(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
	FILE *fp;
	fp = fopen("openclError.err", "w");
	fprintf(fp,"%s\n",errinfo);
	fclose(fp);
	return;
}

size_t nextPow2(size_t num)
{
	size_t temp = num;
	size_t ans = 0;
	if( num > 1)
		ans++;
	while( temp>>=1 )
		ans++;
	if( (1<<(ans-1)) == num)
		ans--;
	return 1<<ans;
}
