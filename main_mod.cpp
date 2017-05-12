/*
Kenneth Hill
STAC-A2 main 
*/
#include "model.h"
#include "option.h"
#include "openclBuilder.h"
#include "openclBuffer.h"
#include "aocl_utils.h"
#include "CL/opencl.h"

#define EPSILON (1e-4f)
#define NUM_THREADS 8192
#define STRING_BUFFER_LEN 1024
#define PRECOMPILED_BINARY "alteraPATHGENnew"
#define RANDSEED 43579
#define KERNEL_UNROLL 8
typedef float hostD;
typedef cl_float kernelD;

using namespace aocl_utils;

void cleanup(void);
void CL_CALLBACK openclError(const char *errinfo, const void *private_info, size_t cb, void *user_data);
size_t nextPow2(size_t num);

int main(int argc, char **argv) {

  cl_uint num_platforms;
  cl_uint num_devices;
   
  #ifndef MODEL_DEBUG
	// TODO I/O from input file
  #endif
  
  input2 = acl_aligned_malloc( sizeof(cl_double)*4*ASSETS*ASSETS);
  if( input2 == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  // correlation matrix
  memcpy(input2, input_A, 4*ASSETS*ASSETS*sizeof(double));

  kappa = acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( kappa == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
printf("Done\n");
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
  
  // get the openCL device ID
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, MAX_DEVICES, &device[0], &num_devices);
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
  //First FPGA//

  cl_ulong memAllocSize_1;
  status = clGetDeviceInfo( device[0], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
	  &memAllocSize_1,NULL);
  if(status != CL_SUCCESS)
  {
	  dump_error("Failed max alloc size",status);
	  return -1;
  }
  printf("Max alloc size: %d MB\n", memAllocSize_1/1024/1024);

  cl_ulong memGlobalSize_1;
  status = clGetDeviceInfo( device[0], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
	  &memGlobalSize_1,NULL);
  if(status != CL_SUCCESS)
  {
	  dump_error("Failed max alloc size",status);
	  return -1;
  }
  printf("Global memory size: %d MB\n", memGlobalSize_1/1024/1024);
  //Second FPGA//  
  cl_ulong memAllocSize_2;
  status = clGetDeviceInfo( device[1], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
	  &memAllocSize_2,NULL);
  if(status != CL_SUCCESS)
  {
	  dump_error("Failed max alloc size",status);
	  return -1;
  }
  printf("Max alloc size: %d MB\n", memAllocSize_2/1024/1024);

  cl_ulong memGlobalSize_2;
  status = clGetDeviceInfo( device[1], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
	  &memGlobalSize_2,NULL);
  if(status != CL_SUCCESS)
  {
	  dump_error("Failed max alloc size",status);
	  return -1;
  }
  printf("Global memory size: %d MB\n", memGlobalSize_2/1024/1024);

  // create a context
  context = clCreateContext(NULL, num_devices, &device[0], &openclError, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateContext.", status);
    cleanup();
    return -1;
  }

  printf("Number of Devices: %d\n",num_devices);
  for(int i =0;i<num_devices;i++)
  {
          //RNG
          queueRNG[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
          if(status != CL_SUCCESS) {
                dump_error("Failed clCreateCommandQueue PATHGEN.", status);
                cleanup();
                return -1;
          }
		  //PATHGEN
          queuePATHGEN[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
          if(status != CL_SUCCESS) {
                dump_error("Failed clCreateCommandQueue PATHGEN.", status);
                cleanup();
                return -1;
          }
  }
  

#ifndef ALTERA_CL  
  // create OpenCL programs
 
 
  if( createProgram("alteraPATHGEN.cl", &programPATHGEN, context, device[0], num_devices) != 0)
  {
	printf("Failed to create programPATHGEN for FPGA # 1");
	cleanup();
	return -1;
  }
  
#else
   //PATHGEN//
  printf("Programming Device(s)\n");

  // Create the program.
  std::string binary_file = getBoardBinaryFile(PRECOMPILED_BINARY, device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  programPATHGEN = createProgramFromBinary(context, binary_file.c_str(), &device[0], num_devices);
  /*
  //PATHGEN
  if( createProgram("alteraPATHGENnewV2.aocx", &programPATHGEN, context, device[0], num_devices) != 0)
  {
	printf("Failed to create programPATHGEN for FPGA #1");
	cleanup();
	return -1;
  }
  */


#endif
  
  // OpenCL kernels
  
  // execution timers
  double common_start,common_end, startF1, endF1, startF2, endF2, common, temp_str, temp_stp;
  // openCL workgroup information
  size_t gSize[3] = { paths,ASSETS, 1};
  size_t gRun[3] = { paths,ASSETS, 1};
  size_t lSize[3] = {1, ASSETS, 1};
  size_t tempSize[3] = {1, ASSETS, 1};
  // openCL workload adjustment
  size_t runN;
  // runtime buffers
  cl_mem *runKernelBuf	= (cl_mem*) malloc(21*sizeof(cl_mem));
  void **runBuffer		= (void**) malloc(21*sizeof(void*));
//  cl_mem *runKernelBuf	= (cl_mem*) malloc(21*sizeof(cl_mem));
//  void **runBuffer		= (void**) malloc(21*sizeof(void*));
  // openCL memReference Count
  cl_uint refCount;
  cl_uint memFree;
  cl_uint flag;
  
#ifdef PLOT
  /*
  cl_mem *runKernelBuf2  = (cl_mem*) malloc(21*sizeof(cl_mem));
 void **runBuffer2       = (void**) malloc(21*sizeof(void*));
 */
 size_t runN2;
 FILE *fp;
#endif

  // kernel timers
  double timeSQRT, timeEXP, timeLOG, timeUNR, timeCORRAND, timePATHGEN, timePATHGEN_t;
 // run MERSERinit
  for(int i=0; i<num_devices;i++) {

//create RNG kernel     
 kernelRNG[i] = clCreateKernel(programPATHGEN, "random_number_generator", &status);
 if(status != CL_SUCCESS){
	dump_error("Failed clCreateKernel alteraPATHGEN.", status);
	return -5;
}

 // create PATHGEN kernel
  kernelPATHGEN[i] = clCreateKernel(programPATHGEN, "alteraPATHGEN", &status);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clCreateKernel alteraPATHGEN.", status);
	return -5;
  }
  }
  
#ifdef PLOT
 // for( int j = 190000; j < 205000; j+=5000)
 	{
	//	int i = 25000;
		int j = atoi(argv[1]);
		paths = j;
		if(num_devices!=1){
		paths = paths/num_devices;
		printf("Total Paths: %d.Dividing Paths into %d. Paths computed per FPGA: %d\n",j, num_devices,paths);
		}
		else{
//		paths = i;
		printf("Paths computed on FPGA: %d\n",paths);
		}
		N             = 2*assets*paths*(unsigned ( hostD (TIMESTEPS) *T));
	//	N2             = 2*assets*paths2*(unsigned ( hostD (TIMESTEPS) *T));
#else
  {
   // paths = 25000;
   // N             = 2*assets*paths*(unsigned ( hostD (TIMESTEPS) *T));
    printf("For paths %d\n", paths);
#endif
  // Configure FPGA 1
  gSize[0] =  gRun[0] = paths;
 // lSize[0] = ASSETS;
  for(int i = 0; i< num_devices;i++) {
  if( clGetKernelWorkGroupInfo( kernelPATHGEN[i], device[i], CL_KERNEL_COMPILE_WORK_GROUP_SIZE, 3*sizeof(size_t), lSize, NULL) != CL_SUCCESS)
  {
	  printf("Failed to find cl workgroup size (1)!\n");
	  cleanup();
	  return -1;
  }  
  if( clGetKernelWorkGroupInfo( kernelRNG[i], device[i], CL_KERNEL_COMPILE_WORK_GROUP_SIZE, 3*sizeof(size_t), tempSize, NULL) != CL_SUCCESS)
  {
	  printf("Failed to find cl workgroup size (1)!\n");
	  cleanup();
	  return -1;
  }  
  }
  openclWork( lSize, gSize, gRun, &runN);
  runN *= 2*totalTimesteps;
  //gRun[0] = gRun[0]/40;
  common_start = timerval();
#ifndef ICDFSET  
  
  
  //kappa//
  runKernelBuf[0]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //invKappa//
  runKernelBuf[1] = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer ICDF",status);
	return -1;
  }
  //theta//
  runKernelBuf[2]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //xi//
  runKernelBuf[3]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //xi2//
  runKernelBuf[4]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //expKD//
  runKernelBuf[5]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //nexpKD//
  runKernelBuf[6]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //nexpKD2//
  runKernelBuf[7]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //S0//
  runKernelBuf[8]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //V0//
  runKernelBuf[9]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //rho//
  runKernelBuf[10]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //invsqrtRho//
  runKernelBuf[11]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //K0//
  runKernelBuf[12]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //K1//
  runKernelBuf[13]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //K2//
  runKernelBuf[14]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //K3//
  runKernelBuf[15]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //K4//
  runKernelBuf[16]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ASSETS, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }	
 //retVolPaths//
  runKernelBuf[17]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*(runN/2), NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
 //retPricePaths//
  runKernelBuf[18]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*(runN/2), NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  /*
 //retRandBuf//
  runKernelBuf[19]  = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*(runN/2), NULL, &status);
  if( status != CL_SUCCESS){
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  */
#else

  //RandSeed//  
  runKernelBuf[0]	= clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(cl_uint), NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer", status);
	return -1;
  }
  //ICDFarray//
  runKernelBuf[1] = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(kernelD)*ICDFSize, NULL, &status);
  if( status != CL_SUCCESS)
  {
	dump_error("Failed clCreateBuffer ICDF",status);
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
#endif
  //FPGA1
  const cl_ulong nr_sims = 10000;
  const cl_ulong total_rnds = nr_sims*(cl_ulong)totalTimesteps*(cl_ulong)NUM_THREADS;
  for(int i=0; i<num_devices; i++) {
#ifndef ICDFSET    
	
	if(initPATHGEN(programPATHGEN, kernelPATHGEN[i], kernelRNG[i], runKernelBuf, 20, delta, totalTimesteps, total_rnds,RANDSEED) != 0)
    {
	  printf("Failed initPATHGEN\n");
	  cleanup();
	  return -1;
    }
#else

    if(initPATHGEN(programPATHGEN, kernelPATHGEN[i], kernelRNG[i], runKernelBuf, 21, delta, totalTimesteps, total_rnds,RANDSEED) != 0)
    {
	  printf("Failed initPATHGEN\n");
	  cleanup();
	  return -1;
    }
#endif
  }
#ifndef ICDFSET

  runBuffer[0]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[0] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[0], kappa, sizeof(kernelD)*ASSETS);
  
  runBuffer[1]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[1] == NULL)
  {
	printf("Host allcation failure\n");
	return -1;
  }
  memcpy( runBuffer[1], invKappa, sizeof(kernelD)*ASSETS);
  
  runBuffer[2]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[2] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  
  memcpy( runBuffer[2], theta, sizeof(kernelD)*ASSETS);
  
  runBuffer[3]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[3] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[3], xi, sizeof(kernelD)*ASSETS);
  
  runBuffer[4]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[4] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[4], xi2, sizeof(kernelD)*ASSETS);
  
  runBuffer[5]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[5] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[5], expKD, sizeof(kernelD)*ASSETS);
  
  runBuffer[6]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[6] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[6], nexpKD, sizeof(kernelD)*ASSETS);
  
  runBuffer[7]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[7] == NULL)
  {
    printf("Host allocation failure\n");
    return -1;	
  }
  memcpy( runBuffer[7], nexpKD2, sizeof(kernelD)*ASSETS);
  
  runBuffer[8]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[8] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[8], S0, sizeof(kernelD)*ASSETS);
  
  runBuffer[9]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[9] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[9], V0, sizeof(kernelD)*ASSETS);
  
  runBuffer[10]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[10] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[10], rho, sizeof(kernelD)*ASSETS);
  
  runBuffer[11]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[11] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[11], invsqrtRho, sizeof(kernelD)*ASSETS);
  
  runBuffer[12]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[12] == 	NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[12], K0, sizeof(kernelD)*ASSETS);
  
  runBuffer[13]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[13] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[13], K1, sizeof(kernelD)*ASSETS);
  
  runBuffer[14]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[14] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[14], K2, sizeof(kernelD)*ASSETS);
  
  runBuffer[15]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[15] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[15], K3, sizeof(kernelD)*ASSETS);
  
  runBuffer[16]		= acl_aligned_malloc( sizeof(kernelD)*ASSETS);
  if( runBuffer[16] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[16], K4, sizeof(kernelD)*ASSETS);
  
  runBuffer[17]		= acl_aligned_malloc( sizeof(kernelD)*(N/2));
  if( runBuffer[17] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  
  runBuffer[18]		= acl_aligned_malloc( sizeof(kernelD)*(N/2));
  if( runBuffer[18] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }

  runBuffer[19]		= acl_aligned_malloc( sizeof(kernelD)*(N/2));
  if( runBuffer[19] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
 
#else

  runBuffer[0]		= acl_aligned_malloc( sizeof(cl_uint));
  if( runBuffer[0] == NULL)
  {
	printf("Host allocation failure\n");
	return -1;
  }
  memcpy( runBuffer[0], (void*) &seed, sizeof(cl_uint));
  
  runBuffer[1]		= acl_aligned_malloc( sizeof(kernelD)*ICDFSize);
  if( runBuffer[1] == NULL)
  {
	printf("Host allcation failure\n");
	return -1;
  }
  ICDFset( (float *) runBuffer[1], "ICDFinput.csv", (unsigned) ICDFSize);

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
#endif
  common_end = timerval();
  common = common_end - common_start;
  
 // startF1 = timerval();
for(int i =0; i< num_devices; i++){
  startF1 = timerval();
#ifndef ICDFSET    
    if( runPATHGEN(gRun, lSize, tempSize, N, queuePATHGEN[i], queueRNG[i], runKernelBuf, runBuffer, 20, kernelPATHGEN[i], kernelRNG[i], &kernel_event[i], &kernel_event_RNG[i], &timePATHGEN) != 0)
  {
	printf("Failed running PATHGEN");
	cleanup();
	return -1;
  }
#else

    if( runPATHGEN(gRun, lSize, tempSize, N, queuePATHGEN[i], queueRNG[i], runKernelBuf, runBuffer, 21, kernelPATHGEN[i], kernelRNG[i], &kernel_event[i], &kernel_event_RNG[i], &timePATHGEN) != 0)
  {
	printf("Failed running PATHGEN");
	cleanup();
	return -1;
  }
#endif
  }
  for(int i=0; i<num_devices; i++) {
#ifndef ICDFSET

    if( get_results(gRun, lSize, tempSize, N, queuePATHGEN[i], queueRNG[i], runKernelBuf, runBuffer, 20, kernelPATHGEN[i], kernelRNG[i], &kernel_event[i], &kernel_event_RNG[i], &timePATHGEN) != 0)

  {
	printf("Failed running PATHGEN");
	cleanup();
	return -1;
  }
#else

    if( get_results(gRun, lSize, tempSize, N, queuePATHGEN[i], queueRNG[i], runKernelBuf, runBuffer, 21, kernelPATHGEN[i], kernelRNG[i], &kernel_event[i], &kernel_event_RNG[i], &timePATHGEN) != 0)
  {
	printf("Failed running PATHGEN");
	cleanup();
	return -1;
  }
#endif
  endF1 = timerval();
  printf("PATHGEN %d execution WALL time: %f\nPATHGEN CL time: %f RNG CL time: %f\n",i , common +(endF1 - startF1), execution_time(kernel_event[i]), execution_time(kernel_event_RNG[i]));
#ifndef NO_OUTPUT
  printf("Generating outputfile from FPGA1\n");
#ifdef PLOT

#ifndef ICDFSET
// writeOutput( (float *) runBuffer[18], (N/2), "PATHGENoutP3.csv");
 //writeOutput( (float *) runBuffer[19], (N/2), "Random.csv");
#else
 writeOutput( (float *) runBuffer[20], (N/2), "PATHGENoutP3.csv");
#endif

#else
  writeOutput( (float *) runBuffer[18], (N/2), "PATHGENoutP1.csv");
#endif
#endif
}
 // endF1 = timerval();
/*
  double diff = endF1-startF1;
  double number_of_sims = (double)nr_sims * (double) totalTimesteps * (double) num_devices;
  printf("Run Time: %.2f\n", diff);
  printf("%d DEVICE ran a total of %lg Simulations\n",num_devices, number_of_sims);
  printf("Throughput = %.2lf Billion Simulations / second\n", number_of_sims/diff);
  */
  //hostD start2,end2;
  //Second FPGA//
  //startF2 = timerval();
#ifdef PLOT
/*
  if(runMERSER(N, queueMERSERinit[1], queueMERSERgen[1], &kernelMERSERinit_t, &kernelMERSERgen_t) != 0)
  {
	printf("Failed running MERSER (2)");
	cleanup();
	return -1;
  
  }
  
  if( runPATHGEN(gRun, lSize, N, queuePATHGEN[1], queueMERSERinit[1], queueMERSERgen[1],  runKernelBuf2, runBuffer2, 21, &kernelPATHGEN_t, &kernelMERSERinit_t, &kernelMERSERgen_t,  &kernel_event_t, &timePATHGEN_t) != 0)
  {
	printf("Failed running PATHGEN");
	cleanup();
	return -1;
  }
#else
  if( runPATHGEN(gRun, lSize, N, queuePATHGEN[1], runKernelBuf, runBuffer, 21, &kernelPATHGEN_t, &kernel_event, &timePATHGEN_t) != 0)
  {
	printf("Failed running PATHGEN");
	cleanup();
	return -1;
  }
  */
#endif
  //endF2 = timerval();
  //printf("PATHGEN 2 execution WALL time: %f\nCL time: %f\n", common + (endF2-startF2), execution_time(kernel_event) );
#ifdef PLOT
 /*
 fp = fopen("TimingKernel2-PCG-hw1.txt","a");
 fprintf(fp,"%d\t%f\t%f\n",paths*2, execution_time(kernel_event[0]), execution_time(kernel_event[1]));
  fclose(fp);
 */
#endif
  
  //printf("PATHGEN execution WALL time: %f\nCL time: %f\n", (end - start) + (end2-start2), execution_time(kernel_event) );
/*
#ifndef NO_OUTPUT
  printf("Generating outputfile from FPGA2\n");
#ifdef PLOT
  writeOutput( (float *) runBuffer2[20], (N/2), "PATHGENoutP3.csv");
#else
  writeOutput( (double *) runBuffer[20], (N/2), "PATHGENoutP2.csv");
#endif
#endif
*/
  // operation cleanup
  //clReleaseKernel( kernelPATHGEN);  
  for( int i=0; i<19; i++)
  {
	clReleaseMemObject(runKernelBuf[i]);
	acl_aligned_free(runBuffer[i]);
  
#ifdef PLOT
    /*
	clReleaseMemObject(runKernelBuf2[i]);
	acl_aligned_free(runBuffer2[i]);
    */
#endif
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
#ifdef PLOT
                /*
		clReleaseMemObject(runKernelBuf2[i]);
		clGetMemObjectInfo(runKernelBuf2[i], CL_MEM_REFERENCE_COUNT, sizeof(cl_uint),
			&refCount, NULL);
                */
#endif
		memFree = i;
	}
  }while(memFree != 20 || flag != 1);
#endif
  }//This is the end of for loop//
#ifdef ALTERA_CL
  // free the resources allocated
  cleanup();
#endif
   return 0;
}

// free the resources allocated during initialization
void cleanup(void) {
    if(programPATHGEN)
	clReleaseProgram(programPATHGEN);
    if(context)
        clReleaseContext(context);
for(int i=0; i<MAX_DEVICES; i++)
{
 //RNG
 if(kernel_event_RNG[i])
        clReleaseEvent(kernel_event_RNG[i]);
 if(kernelRNG[i])
	clReleaseKernel(kernelRNG[i]);
 if(queueRNG[i])
        clReleaseCommandQueue(queueRNG[i]);
 //PATHGEN
 if(kernel_event[i])
        clReleaseEvent(kernel_event[i]);
 if(kernelPATHGEN[i])
	clReleaseKernel(kernelPATHGEN[i]);
 if(queuePATHGEN[i])
        clReleaseCommandQueue(queuePATHGEN[i]);
 
}
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
