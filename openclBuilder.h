#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

//#include "model.h"
#include "main.h"
#include "util.h"
#include "openclBuffer.h"

// run OpenCL kernels
int get_results(size_t *globalSize, size_t *localSize, size_t *tempSize, size_t N, cl_command_queue queue1, cl_command_queue queue2, cl_mem *kernelBuffers, void **buffers,unsigned bufferSize, cl_kernel kernelPATHGEN, cl_kernel kernelRNG, cl_event *kernel_event, cl_event *kernel_event_RNG, double *time);
int runPATHGEN(size_t *globalSize, size_t *localSize, size_t *tempSize, size_t N, cl_command_queue queue1, cl_command_queue queue2, cl_mem *kernelBuffers, void **buffers,unsigned bufferSize, cl_kernel kernelPATHGEN, cl_kernel kernelRNG, cl_event *kernel_event, cl_event *kernel_event_RNG, double *time);

// create OpenCL buffers
int initKernelBuffers(size_t ICDFSize, size_t N, cl_context context, cl_mem **buffers, unsigned bufferSize);
// create OpenCL kernels
int initMERSER(ulong N, cl_kernel *kernelMERSERgen);
int initPATHGEN(cl_program programPATHGEN, cl_kernel kernelPATHGEN, cl_kernel kernelRNG, cl_mem *buffers, unsigned bufferSize, double delta, unsigned totalTimesteps, ulong N, unsigned int randSEED);
//int initPATHGEN(cl_program programPATHGEN, cl_kernel kernelPATHGEN, cl_kernel kernelMERSERgen,  cl_mem *buffers, unsigned bufferSize, double delta, unsigned totalTimesteps, ulong N, cl_mem *randKernelBuf);
// create OpenCL programs
int createProgram(char *source_file, cl_program *retProgram, cl_context context, cl_device_id device, cl_uint num_devices);
