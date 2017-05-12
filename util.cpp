#include "util.h"


char* load_program(const char* file_name, size_t* length)
{
    FILE* f_stream = NULL;
    size_t size;

    #ifdef _WIN32
        if(fopen_s(&f_stream, file_name, "rb") != 0) 
        {       
            return NULL;
        }
    #else
        f_stream = fopen(file_name, "rb");
        if(f_stream == 0) 
        {       
            return NULL;
        }
    #endif

    // Get file size
    fseek(f_stream, 0, SEEK_END); 
    size = ftell(f_stream);
    fseek(f_stream, 0, SEEK_SET); 

    // Allocate buffer for source
    char* cl_source = (char *)malloc(size + 1); 

	// Read file
    if (fread(cl_source, size, 1, f_stream) != 1)
    {
        fclose(f_stream);
        free(cl_source);
        return NULL;
    }

    // Close file and return its size to caller
    fclose(f_stream);
    if(length != NULL)
    {
        *length = size;
    }
	// Finish a string with a \0.
    cl_source[size] = '\0';

	// Return reference to array to caller.
    return cl_source;
}

double execution_time(cl_event &event)
{
	cl_ulong start, end;
    
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    
	return (double)1.0e-9 * (end - start);
}

void dump_error(const char *str, cl_int status) {
	printf("%s\n", str);
	printf("Error code: %d\n", status);
}

// Set lookup table for Gaussian RNG
void ICDFset( float *ICDF, char *coefFile, unsigned size)
{
	FILE *fp;
#ifdef LINUX
	fp = fopen(coefFile,"r");
#else
	fopen_s(&fp,coefFile,"r");
#endif
	if(fp==NULL)
	{
		printf("FILE I/O Error\n");
	}

	fseek( fp,0L, SEEK_SET);

	char temp[100];
	unsigned index = 0;

	while( index<size )
	{
#ifdef LINUX
		if(fscanf(fp,"%s", temp) == EOF)
		{
#else
		if(fscanf_s(fp, "%s", temp, 50) == EOF )
        {
#endif
			break;
        } else{
//			printf("String %s\n",temp);
			ICDF[index] = atof(temp);
//			printf("ICDF %i: %f\n",index,ICDF[index]);
            index++;
        }
	}

	fclose(fp);
	return;
}

void writeOutput( float *data, size_t size, const char *fileName )
{
	FILE *fp;
#ifdef LINUX	

#ifdef PLOT
	fp = fopen(fileName,"a");
#else
  fp = fopen(fileName,"w");
#endif
#else
	fopen_s(&fp,fileName,"w");
#endif
	if(fp==NULL)
	{
        printf("FILE I/O Error for %s\n",fileName);
		return;
	}

	fseek( fp,0L, SEEK_SET);

	for( size_t i=0; i<size; i++  )
	{
		fprintf( fp,"%f\n", data[i] );
	}

	fclose(fp);
	return;
}

// Need to align host data to 32 bytes to be able to use DMA
// LINUX/WINDOWS macros are defined in Makefiles.
#define ACL_ALIGNMENT 32

#ifdef LINUX
	#include <stdlib.h>
	void* acl_aligned_malloc (size_t size) {
	  void *result = NULL;
	  posix_memalign (&result, ACL_ALIGNMENT, size);
	  return result;
	}
	void acl_aligned_free (void *ptr) {
	  free (ptr);
	}
#else // WINDOWS
	void* acl_aligned_malloc (size_t size) {
	  return _aligned_malloc (size, ACL_ALIGNMENT);
	}
	void acl_aligned_free (void *ptr) {
	  _aligned_free (ptr);
	}
#endif // LINUX

#ifdef LINUX
  //do nothing if already defined
#else
 
  //fix Windows trying to reinvent the wheel
 
  int gettimeofday(struct timeval *tv, struct timezone *tz)
  {
  // Define a structure to receive the current Windows filetime
    FILETIME ft;
 
  // Initialize the present time to 0 and the timezone to UTC
    unsigned __int64 tmpres = 0;
    static int tzflag = 0;
 
    if (NULL != tv)
    {
      GetSystemTimeAsFileTime(&ft);
 
  // The GetSystemTimeAsFileTime returns the number of 100 nanosecond
  // intervals since Jan 1, 1601 in a structure. Copy the high bits to
  // the 64 bit tmpres, shift it left by 32 then or in the low 32 bits.
      tmpres |= ft.dwHighDateTime;
     tmpres <<= 32;
     tmpres |= ft.dwLowDateTime;

 // Convert to microseconds by dividing by 10
     tmpres /= 10;

 // The Unix epoch starts on Jan 1 1970.  Need to subtract the difference
 // in seconds from Jan 1 1601.
     tmpres -= DELTA_EPOCH_IN_MICROSECS;

 // Finally change microseconds to seconds and place in the seconds value.
 // The modulus picks up the microseconds.
     tv->tv_sec = (long)(tmpres / 1000000UL);
     tv->tv_usec = (long)(tmpres % 1000000UL);
   }

   if (NULL != tz)
   {
     if (!tzflag)
     {
       _tzset();
       tzflag++;
     }

 // Adjust for the timezone west of Greenwich
       tz->tz_minuteswest = _timezone / 60;
     tz->tz_dsttime = _daylight;
   }

   return 0;
}
#endif
/*
double timerval(void)
{
	struct timeval st;
	gettimeofday(&st,NULL);
	return st.tv_sec + st.tv_usec*1e-6;
}
*/
void openclWork( size_t *lSize, size_t *gSize, size_t *gRun, size_t *runN)
{
	for( int i=0; i<3; i++)
	{
		if( gSize[i]%lSize[i] == 0)
		{
			gRun[i] = gSize[i];
		} else
		{
			gRun[i] = lSize[i] - (gSize[i]%lSize[i]) + gSize[i];
		}
	}
	*runN = gRun[0]*gRun[1]*gRun[2];
	return;
}

int squareMatrixInput( void *dest, void *src, size_t *size, size_t *adjSize)
{
	double *tdest, *tsrc;
	//paths
	for( int i=0; i<size[2]; i++)
	{
		//processes
		for( int j=0; j<size[0]; j++)
		{
			tdest = &((double*)dest)[ (i*size[0]*adjSize[1] + j*adjSize[1])];
			tsrc = &((double*)src)[ (i*size[0]*size[1] + j*size[1])];
			memcpy( tdest, tsrc, sizeof(cl_double)*size[1]);
			for( int k=0; k<(adjSize[1] - size[1]); k++)
			{
				((double *)dest)[ i*size[0]*adjSize[1] + j*adjSize[1] + size[1] + k] = 0.0;
			}
		}
	}
	return 0;
}

int unsquareMatrixOutput( void *dest, void *src, size_t *size, size_t *adjSize)
{
	double *tdest, *tsrc;
	//paths
	for( int i=0; i<size[2]; i++)
	{
		//processes
		for( int j=0; j<size[0]; j++)
		{
			tdest = &((double*)dest)[ (i*size[0]*size[1] + j*size[1])];
			tsrc = &((double*)src)[ (i*size[0]*adjSize[1] + j*adjSize[1])];
			memcpy( tdest, tsrc, sizeof(cl_double)*size[1]);
		}
	}
	return 0;
}
