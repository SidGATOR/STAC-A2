#include "openclBuilder.h"


typedef cl_float kernelD;

// Enqueue PATHGEN kernelnt runPATHGEN(size_t *globalSize, size_t *localSize, size_t N, cl_command_queue queue, cl_mem *kernelBuffers, void **buffers,

int runPATHGEN(size_t *globalSize, size_t *localSize, size_t *tempSize, size_t N, cl_command_queue queue1, cl_command_queue queue2, cl_mem *kernelBuffers, void **buffers, unsigned bufferSize, cl_kernel kernelPATHGEN, cl_kernel kernelRNG,  cl_event *kernel_event, cl_event *kernel_event_RNG, double *time)
{

#ifndef ICDFSET
	if(bufferSize != 20)
	{
		printf("runPATHGEN: incorrect number of buffers\n");
		return -5;
	}
	cl_int status;
  //int ICDFSize = 8192;
  /*
	// send Gaussian random numbers (kernel_rnArray)
	status = clEnqueueWriteBuffer( queue, kernelBuffers[0], CL_FALSE, 0, sizeof(cl_uint), buffers[0], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue Gaussian random numbers.",status);
		return -1;
	}

	// send uniform random numbers (kernel_unArray)
	status = clEnqueueWriteBuffer( queue, kernelBuffers[1], CL_FALSE, 0, sizeof(kernelD)*8192, buffers[1], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue Uniform random numbers.",status);
		return -1;
	}
	*/

	// send kappa values (kernel_kappa)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[0], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[0], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue kappa values.",status);
		return -1;
	}

	// send inverse kappa values (kernel_invKappa)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[1], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[1], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue inverse kappa.",status);
		return -1;
	}

	// send theta values (kernel_theta)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[2], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[2], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue theta.",status);
		return -1;
	}

	// send xi (kernel_xi)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[3], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[3], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue xi.",status);
		return -1;
	}

	// send xi2 (kernel_xi2)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[4], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[4], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue xi2.",status);
		return -1;
	}

	// send expKD (kernel_expKD)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[5], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[5], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue expKD.",status);
		return -1;
	}

	// send nexpKD (kernel_nexpKD)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[6], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[6], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue nexpKD.",status);
		return -1;
	}

	// send nexpKD2 (kernel_nexpKD2)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[7], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[7], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue nexpKD2.",status);
		return -1;
	}

	// send S0 (kernel_S0)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[8], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[8], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue S0.",status);
		return -1;
	}

	// send V0 (kernel_V0)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[9], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[9], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue V0.",status);
		return -1;
	}
	
	// send rho (kernel_rho)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[10], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[10], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue rho.",status);
		return -1;
	}
	
	// send invsqrtRho (kernel_invsqrtRho)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[11], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[11], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue invsqrtRho.",status);
		return -1;
	}
	
	// send K0 (kernel_K0)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[12], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[12], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue K0.",status);
		return -1;
	}
	
	// send K1 (kernel_K1)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[13], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[13], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue K1.",status);
		return -1;
	}
	
	// send K2 (kernel_K2)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[14], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[14], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue K2.",status);
		return -1;
	}
	
	// send K3 (kernel_K3)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[15], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[15], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue K3.",status);
		return -1;
	}
	
	// send K4 (kernel_K4)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[16], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[16], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue K4.",status);
		return -1;
	}
//	int temp = globalSize[0];
//	globalSize[0] = globalSize[0]/2;
//	globalSize[2] = 252;
   // 1. Launching RNG
   status = clEnqueueNDRangeKernel(queue2,kernelRNG,2, NULL, globalSize, tempSize, 0, NULL, kernel_event_RNG); 
	if(status != CL_SUCCESS)
	{
		dump_error("Failed to launch kernelRNG.", status);
		if(status == CL_INVALID_WORK_GROUP_SIZE)
		{
			printf("test fail\n");
		}
		return -10;
	}
//	globalSize[2] = 1;
 //  globalSize[0] = temp;
   // 2.Launching PATHGEN 
	status = clEnqueueNDRangeKernel(queue1, kernelPATHGEN, 2, NULL, globalSize, localSize, 0, NULL, kernel_event);
	if(status != CL_SUCCESS)
	{
		dump_error("Failed to launch kernelPATHGEN.", status);
		if(status == CL_INVALID_WORK_GROUP_SIZE)
		{
			printf("test\n");
		}
		return -10;
	}
	
	printf("localSize: %d\t%d\t%d\n",localSize[0],localSize[1],localSize[2]);
	printf("tempSize: %d\t%d\t%d\n",tempSize[0],tempSize[1],tempSize[2]);
	
	status = clFlush(queue2);
	status |= clFlush(queue1);
	if(status != CL_SUCCESS)
	{
		dump_error("Failed clFlush", status);
		return -100;
	}
	return 0;
#else

	if(bufferSize != 21)
	{
		printf("runPATHGEN: incorrect number of buffers\n");
		return -5;
	}
	cl_int status;
  //int ICDFSize = 8192;
  
	// send Gaussian random numbers (kernel_rnArray)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[0], CL_FALSE, 0, sizeof(cl_uint), buffers[0], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue Gaussian random numbers.",status);
		return -1;
	}

	// send uniform random numbers (kernel_unArray)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[1], CL_FALSE, 0, sizeof(kernelD)*8192, buffers[1], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue Uniform random numbers.",status);
		return -1;
	}
	

	// send kappa values (kernel_kappa)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[2], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[2], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue kappa values.",status);
		return -1;
	}

	// send inverse kappa values (kernel_invKappa)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[3], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[3], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue inverse kappa.",status);
		return -1;
	}

	// send theta values (kernel_theta)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[4], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[4], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue theta.",status);
		return -1;
	}

	// send xi (kernel_xi)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[5], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[5], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue xi.",status);
		return -1;
	}

	// send xi2 (kernel_xi2)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[6], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[6], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue xi2.",status);
		return -1;
	}

	// send expKD (kernel_expKD)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[7], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[7], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue expKD.",status);
		return -1;
	}

	// send nexpKD (kernel_nexpKD)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[8], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[8], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue nexpKD.",status);
		return -1;
	}

	// send nexpKD2 (kernel_nexpKD2)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[9], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[9], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue nexpKD2.",status);
		return -1;
	}

	// send S0 (kernel_S0)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[10], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[10], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue S0.",status);
		return -1;
	}

	// send V0 (kernel_V0)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[11], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[11], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue V0.",status);
		return -1;
	}
	
	// send rho (kernel_rho)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[12], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[12], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue rho.",status);
		return -1;
	}
	
	// send invsqrtRho (kernel_invsqrtRho)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[13], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[13], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue invsqrtRho.",status);
		return -1;
	}
	
	// send K0 (kernel_K0)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[14], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[14], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue K0.",status);
		return -1;
	}
	
	// send K1 (kernel_K1)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[15], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[15], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue K1.",status);
		return -1;
	}
	
	// send K2 (kernel_K2)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[16], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[16], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue K2.",status);
		return -1;
	}
	
	// send K3 (kernel_K3)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[17], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[17], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue K3.",status);
		return -1;
	}
	
	// send K4 (kernel_K4)
	status = clEnqueueWriteBuffer( queue1, kernelBuffers[18], CL_FALSE, 0, sizeof(kernelD)*ASSETS, buffers[18], 0, NULL, NULL);
	if( status != CL_SUCCESS)
	{
		dump_error("PATHGEN failed to enqueue K4.",status);
		return -1;
	}
   // 1. Launching RNG
   status = clEnqueueNDRangeKernel(queue2,kernelRNG,2, NULL, globalSize, tempSize, 0, NULL, kernel_event_RNG); 
	if(status != CL_SUCCESS)
	{
		dump_error("Failed to launch kernelRNG.", status);
		if(status == CL_INVALID_WORK_GROUP_SIZE)
		{
			printf("test fail\n");
		}
		return -10;
	}
   // 2.Launching PATHGEN 
	status = clEnqueueNDRangeKernel(queue1, kernelPATHGEN, 2, NULL, globalSize, localSize, 0, NULL, kernel_event);
	if(status != CL_SUCCESS)
	{
		dump_error("Failed to launch kernelPATHGEN.", status);
		if(status == CL_INVALID_WORK_GROUP_SIZE)
		{
			printf("test\n");
		}
		return -10;
	}
	status = clFlush(queue2);
	status |= clFlush(queue1);
	if(status != CL_SUCCESS)
	{
		dump_error("Failed clFlush", status);
		return -100;
	}
	return 0;
#endif
}

int get_results(size_t *globalSize, size_t *localSize, size_t *tempSize, size_t N, cl_command_queue queue1, cl_command_queue queue2, cl_mem *kernelBuffers, void **buffers, unsigned bufferSize, cl_kernel kernelPATHGEN, cl_kernel kernelRNG,  cl_event *kernel_event, cl_event *kernel_event_RNG, double *time)
{
#ifndef ICDFSET
	cl_int status;
	/*
	// read the random numbers
	status = clEnqueueReadBuffer(queue2, kernelBuffers[19], CL_TRUE, 0, sizeof(kernelD) * (N/2), buffers[19], 0, NULL, NULL);
	if(status != CL_SUCCESS)
	{
		dump_error("Failed to read output volPaths for PATHGEN.", status);
		return -10;
	}

	// read the output (kernel_retVolPaths)
	status = clEnqueueReadBuffer(queue1, kernelBuffers[17], CL_TRUE, 0, sizeof(kernelD) * (N/2), buffers[17], 0, NULL, NULL);
	if(status != CL_SUCCESS)
	{
		dump_error("Failed to read output volPaths for PATHGEN.", status);
		return -10;
	}
*/
	// read the output (kernel_retPricPaths)
	status = clEnqueueReadBuffer(queue1, kernelBuffers[18], CL_TRUE, 0, sizeof(kernelD) * (N/2), buffers[18], 0, NULL, NULL);
	if(status != CL_SUCCESS)
	{
		dump_error("Failed to read output pricePaths for PATHGEN.", status);
		return -10;
	}
/*	
    //read the random number (kernel_retRandBuf)
	status = clEnqueueReadBuffer(queue, kernelBuffers[19], CL_TRUE, 0 , sizeof(kernelD) * (N/2), buffers[19],0 , NULL, NULL);
	if(status != CL_SUCCESS){
		
		dump_error("Failed to read output random numbers for PATHGEN.",status);
		return -10;
	}
*/
	return 0;	
#else
	
	cl_int status;
	// read the output (kernel_retVolPaths)
	status = clEnqueueReadBuffer(queue1, kernelBuffers[19], CL_TRUE, 0, sizeof(kernelD) * (N/2), buffers[19], 0, NULL, NULL);
	if(status != CL_SUCCESS)
	{
		dump_error("Failed to read output volPaths for PATHGEN.", status);
		return -10;
	}

	// read the output (kernel_retPricPaths)
	status = clEnqueueReadBuffer(queue1, kernelBuffers[20], CL_TRUE, 0, sizeof(kernelD) * (N/2), buffers[20], 0, NULL, NULL);
	if(status != CL_SUCCESS)
	{
		dump_error("Failed to read output pricePaths for PATHGEN.", status);
		return -10;
	}

	return 0;
#endif
}

int initPATHGEN(cl_program programPATHGEN, cl_kernel kernelPATHGEN, cl_kernel kernelRNG, cl_mem *buffers, unsigned bufferSize, double delta, unsigned totalTimesteps, ulong N, unsigned int randSEED)
//int initPATHGEN(cl_program programPATHGEN, cl_kernel kernelPATHGEN, cl_kernel kernelMERSERgen,  cl_mem *buffers, unsigned bufferSize, double delta, unsigned totalTimesteps, ulong N, cl_mem *randKernelBuf)
{
#ifndef ICDFSET
  if(bufferSize != 20)
  {
	  printf("PATHGEN incorrect number of buffers\n");
	  return -5;
  }
  cl_int status;
  
  // set delta
  status = clSetKernelArg(kernelPATHGEN, 0, sizeof(cl_float), (void*) &delta);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg 0 of PATHGEN.", status);
	return -15;
  }
  
   // set timesteps
  status = clSetKernelArg(kernelPATHGEN, 1, sizeof(cl_uint), (void*) &totalTimesteps);
  if(status != CL_SUCCESS) 
  {
	dump_error("Failed clSetKernelArg 1 of PATHGEN.", status);
	return -15;
  }
  // set kappa (kernel_kappa)
  status = clSetKernelArg(kernelPATHGEN, 2, sizeof(cl_mem), (void*) &buffers[0]);
  if(status != 	CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg kappa of PATHGEN.", status);
	return -15;
  }
  
  // set invKappa (kernel_invKappa)
  status = clSetKernelArg(kernelPATHGEN, 3, sizeof(cl_mem), (void*) &buffers[1]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg invKappa of PATHGEN.", status);
	return -15;
  }
  
  // set theta (kernel_theta)
  status = clSetKernelArg(kernelPATHGEN, 4, sizeof(cl_mem), (void*) &buffers[2]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg theta of PATHGEN.", status);
	return -15;
  }
  
  // set xi (kernel_xi)
  status = clSetKernelArg(kernelPATHGEN, 5, sizeof(cl_mem), (void*) &buffers[3]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg xi of PATHGEN.", status);
	return -15;
  }
  
  // set xi2 (kernel_xi2)
  status = clSetKernelArg(kernelPATHGEN, 6, sizeof(cl_mem), (void*) &buffers[4]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg xi2 of PATHGEN.", status);
	return -15;
  }
  
  // set expKD (kernel_expKD)
  status = clSetKernelArg(kernelPATHGEN, 7, sizeof(cl_mem), (void*) &buffers[5]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg expKD of PATHGEN.", status);
	return -15;
  }
  
  // set nexpKD (kernel_nexpKD)
  status = clSetKernelArg(kernelPATHGEN, 8, sizeof(cl_mem), (void*) &buffers[6]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg nexpKD of PATHGEN.", status);
	return -15;
  }	
  
  // set nexpKD2 (kernel_nexpKD2)
  status = clSetKernelArg(kernelPATHGEN, 9, sizeof(cl_mem), (void*) &buffers[7]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg nexpKD2 of PATHGEN.", status);
	return -15;
  }
  
  // set S0 (kernel_S0)
  status = clSetKernelArg(kernelPATHGEN, 10, sizeof(cl_mem), (void*) &buffers[8]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg S0 of PATHGEN.", status);
	return -15;
  }
  
  // set V0 (kernel_V0)
  status = clSetKernelArg(kernelPATHGEN, 11, sizeof(cl_mem), (void*) &buffers[9]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg V0 of PATHGEN.", status);
	return -15;
  }
  
  // set rho (kernel_rho)
  status = clSetKernelArg(kernelPATHGEN, 12, sizeof(cl_mem), (void*) &buffers[10]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg rho of PATHGEN.", status);
	return -15;
  }
  
  // set invsqrtRho (kernel_invsqrtRho)
  status = clSetKernelArg(kernelPATHGEN, 13, sizeof(cl_mem), (void*) &buffers[11]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg invsqrtRho of PATHGEN.", status);
	return -15;
  }
  
  // set K0 (kernel_K0)
  status = clSetKernelArg(kernelPATHGEN, 14, sizeof(cl_mem), (void*) &buffers[12]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg K0 of PATHGEN.", status);
	return -15;
  }
  
  //set K1 (kernel_K1)
  status = clSetKernelArg(kernelPATHGEN, 15, sizeof(cl_mem), (void*) &buffers[13]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg K1 of PATHGEN.", status);
	return -15;
  }
  
  // set K2 (kernel_K2)
  status = clSetKernelArg(kernelPATHGEN, 16, sizeof(cl_mem), (void*) &buffers[14]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg K2 if PATHGEN.", status);
	return -15;
  }
  
  // set K3 (kernel_K3) 
  status = clSetKernelArg(kernelPATHGEN, 17, sizeof(cl_mem), (void*) &buffers[15]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg K3 of PATHGEN.", status);
	return -15;
  }
  
  // set K4 (kernel_K4)
  status = clSetKernelArg(kernelPATHGEN, 18, sizeof(cl_mem), (void*) &buffers[16]);
  if(status != CL_SUCCESS) 
  {
	dump_error("Failed clSetKernelArg K4 of PATHGEN.", status);
	return -15;
  }
  /*
  // set retVolPaths (kernel_retVolPaths)
  status = clSetKernelArg(kernelPATHGEN, 19, sizeof(cl_mem), (void*) &buffers[17]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg retVolPaths of PATHGEN.", status);
	return -15;
  }
  */
  // set retPricePaths (kernel_retPricePaths)
  status = clSetKernelArg(kernelPATHGEN, 19, sizeof(cl_mem), (void*) &buffers[18]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg retPricePaths of PATHGEN.", status);
	return -15;
  }
 /* 
  //Random Buffer//
  status = clSetKernelArg(kernelRNG, 0, sizeof(cl_mem), (void*) &buffers[19]);
  
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg retPricePaths of PATHGEN.", status);
	return -15;
  }
 */
#else

  if(bufferSize != 21)
  {
	  printf("PATHGEN incorrect number of buffers\n");
	  return -5;
  }
  cl_int status;
  
  // set delta
  status = clSetKernelArg(kernelPATHGEN, 0, sizeof(cl_float), (void*) &delta);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg 0 of PATHGEN.", status);
	return -15;
  }
  
   // set timesteps
  status = clSetKernelArg(kernelPATHGEN, 1, sizeof(cl_uint), (void*) &totalTimesteps);
  if(status != CL_SUCCESS) 
  {
	dump_error("Failed clSetKernelArg 1 of PATHGEN.", status);
	return -15;
  }
  
  // set Random Seed (kernel_seed)
  status = clSetKernelArg(kernelPATHGEN, 2, sizeof(cl_uint), (void*) &buffers[0]);
  if(status != 	CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg kappa of PATHGEN.", status);
	return -15;
  }
  
  // set ICDF Array (kernel_ICDF)
  status = clSetKernelArg(kernelPATHGEN, 3, sizeof(cl_mem), (void*) &buffers[1]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg invKappa of PATHGEN.", status);
	return -15;
  }
  // set kappa (kernel_kappa)
  status = clSetKernelArg(kernelPATHGEN, 4, sizeof(cl_mem), (void*) &buffers[2]);
  if(status != 	CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg kappa of PATHGEN.", status);
	return -15;
  }
  
  // set invKappa (kernel_invKappa)
  status = clSetKernelArg(kernelPATHGEN, 5, sizeof(cl_mem), (void*) &buffers[3]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg invKappa of PATHGEN.", status);
	return -15;
  }
  
  // set theta (kernel_theta)
  status = clSetKernelArg(kernelPATHGEN, 6, sizeof(cl_mem), (void*) &buffers[4]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg theta of PATHGEN.", status);
	return -15;
  }
  
  // set xi (kernel_xi)
  status = clSetKernelArg(kernelPATHGEN, 7, sizeof(cl_mem), (void*) &buffers[5]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg xi of PATHGEN.", status);
	return -15;
  }
  
  // set xi2 (kernel_xi2)
  status = clSetKernelArg(kernelPATHGEN, 8, sizeof(cl_mem), (void*) &buffers[6]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg xi2 of PATHGEN.", status);
	return -15;
  }
  
  // set expKD (kernel_expKD)
  status = clSetKernelArg(kernelPATHGEN, 9, sizeof(cl_mem), (void*) &buffers[7]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg expKD of PATHGEN.", status);
	return -15;
  }
  
  // set nexpKD (kernel_nexpKD)
  status = clSetKernelArg(kernelPATHGEN, 10, sizeof(cl_mem), (void*) &buffers[8]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg nexpKD of PATHGEN.", status);
	return -15;
  }	
  
  // set nexpKD2 (kernel_nexpKD2)
  status = clSetKernelArg(kernelPATHGEN, 11, sizeof(cl_mem), (void*) &buffers[9]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg nexpKD2 of PATHGEN.", status);
	return -15;
  }
  
  // set S0 (kernel_S0)
  status = clSetKernelArg(kernelPATHGEN, 12, sizeof(cl_mem), (void*) &buffers[10]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg S0 of PATHGEN.", status);
	return -15;
  }
  
  // set V0 (kernel_V0)
  status = clSetKernelArg(kernelPATHGEN, 13, sizeof(cl_mem), (void*) &buffers[11]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg V0 of PATHGEN.", status);
	return -15;
  }
  
  // set rho (kernel_rho)
  status = clSetKernelArg(kernelPATHGEN, 14, sizeof(cl_mem), (void*) &buffers[12]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg rho of PATHGEN.", status);
	return -15;
  }
  
  // set invsqrtRho (kernel_invsqrtRho)
  status = clSetKernelArg(kernelPATHGEN, 15, sizeof(cl_mem), (void*) &buffers[13]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg invsqrtRho of PATHGEN.", status);
	return -15;
  }
  
  // set K0 (kernel_K0)
  status = clSetKernelArg(kernelPATHGEN, 16, sizeof(cl_mem), (void*) &buffers[14]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg K0 of PATHGEN.", status);
	return -15;
  }
  
  //set K1 (kernel_K1)
  status = clSetKernelArg(kernelPATHGEN, 17, sizeof(cl_mem), (void*) &buffers[15]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg K1 of PATHGEN.", status);
	return -15;
  }
  
  // set K2 (kernel_K2)
  status = clSetKernelArg(kernelPATHGEN, 18, sizeof(cl_mem), (void*) &buffers[16]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg K2 if PATHGEN.", status);
	return -15;
  }
  
  // set K3 (kernel_K3) 
  status = clSetKernelArg(kernelPATHGEN, 19, sizeof(cl_mem), (void*) &buffers[17]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg K3 of PATHGEN.", status);
	return -15;
  }
  
  // set K4 (kernel_K4)
  status = clSetKernelArg(kernelPATHGEN, 20, sizeof(cl_mem), (void*) &buffers[18]);
  if(status != CL_SUCCESS) 
  {
	dump_error("Failed clSetKernelArg K4 of PATHGEN.", status);
	return -15;
  }
  
  // set retVolPaths (kernel_retVolPaths)
  status = clSetKernelArg(kernelPATHGEN, 21, sizeof(cl_mem), (void*) &buffers[19]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg retVolPaths of PATHGEN.", status);
	return -15;
  }
  
  // set retPricePaths (kernel_retPricePaths)
  status = clSetKernelArg(kernelPATHGEN, 22, sizeof(cl_mem), (void*) &buffers[20]);
  if(status != CL_SUCCESS)
  {
	dump_error("Failed clSetKernelArg retPricePaths of PATHGEN.", status);
	return -15;
  }
#endif
  return 0;
  
}

int createProgram(char *source_file, cl_program *retProgram, cl_context context, cl_device_id device, cl_uint num_devices)
{
  cl_int status;
  // Read program file
  size_t program_length;
  char *source = load_program(source_file, &program_length);
  if(NULL == source)
  {
    printf("ERROR: Failed to load program source!\n");
    return -10000;
  }
    
#ifndef ALTERA_CL

  // create the program
  *retProgram = clCreateProgramWithSource(context, num_devices, (const char **)&source, &program_length, &status);
  if (status != CL_SUCCESS)
  {
    printf("ERROR: Failed to create a program\n");
	return status;
  }
#else

  // create the kernel
  cl_int kernel_status;
  *retProgram = clCreateProgramWithBinary(context, num_devices, &device, &program_length, (const unsigned char**)&source, 
	&kernel_status, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateProgramWithBinary.", status);
    return -25;
  }

#endif

  free(source);
  // build the program
  status = clBuildProgram(*retProgram, num_devices, &device, "", NULL, NULL);
  if (status != CL_SUCCESS)
  {
    printf("ERROR: Unable to build program!\n");
    return -20;
  }

  return 0;
  
}
