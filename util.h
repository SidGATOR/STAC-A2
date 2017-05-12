#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CL/opencl.h"


 #ifdef LINUX
         #include <sys/time.h>
 #else
 // Define timer for Windows machines
         #include <time.h>
        #include <windows.h>
         #if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
                 #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
         #else
                 #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
         #endif

 struct timezone
 {
   int  tz_minuteswest; /* minutes W of Greenwich */
   int  tz_dsttime;     /* type of dst correction */
 };


 // Definition of a gettimeofday function

extern int gettimeofday(struct timeval *tv, struct timezone *tz);
#endif

// utility functions
char* load_program(const char* file_name, size_t* length);
double execution_time(cl_event &event);
void dump_error(const char *str, cl_int status);
void ICDFset( float *ICDF, char *coefFile, unsigned size);
void writeOutput( float *data, size_t size, const char *fileName );
void* acl_aligned_malloc (size_t size);
void acl_aligned_free (void *ptr);
double timerval(void);
void openclWork( size_t *lSize, size_t *gSize, size_t *gRun, size_t *runN);
int squareMatrixInput( void *dest, void *src, size_t *size, size_t *adjSize);
int unsquareMatrixOutput( void *dest, void *src, size_t *size, size_t *adjSize);
