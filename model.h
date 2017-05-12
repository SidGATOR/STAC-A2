#include <math.h>
#include "main.h"
#include "CL/opencl.h"

// Heston model parameters
#define  MAX_DEVICES 2

#ifdef MODEL_DEBUG
	// risk-free rate
	static double r					= 0.02;
	// fixed strike price
	static double K					= 1.0;
	// contract maturity
	static double T					= 1.0;
	// number of simulation paths
	static unsigned paths			= 25000;
	// underlyings in contract
	static unsigned assets 			= 5;
	// Hestom kappa parameters
	static double input_kappa[5]	= { 1.49394, 2.49858, 8.26597, 7.37047, 3.6747};
	// Heston theta parameters
	static double input_theta[5]	= { 0.014657, 0.113096, 3.24965, 0.122812, 0.0480654 };
	// Heston xi parameters
	static double input_xi[5]		= { 0.54534, 0.76203, 0.685527, 0.552243, 0.844805 };
	// correlation between underlyings
	static double input_rho[5]		= { 0.0256317, 0.260682, 0.0557182, 0.0111027, 0.294668 };
	// Initial underlying volatility
	static double input_V0[5]		= { 0.676705, 0.831754, 0.442274, 0.0747174, 0.120731 };
	// Initial underlying value
	static double input_S0[5]		= { 1.0, 1.0, 1.0, 1.0, 1.0 };
	// log-space price
	static double input_Y0[5]		= { log(1.0), log(1.0), log(1.0), log(1.0), log(1.0) };
	// corr matirix
	static double input_A[100]		= { 1.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.000000, 0.00000, 0.0000, 0.0000, 0.0000,
										-0.15446, 0.98800, 0.00000, 0.00000, 0.00000, 0.000000, 0.00000, 0.0000, 0.0000, 0.0000,
										0.76188, 0.01423, 0.64756, 0.00000, 0.00000, 0.000000, 0.00000, 0.0000, 0.0000, 0.0000,
										0.39599, 0.24597, -0.05864, 0.88275, 0.00000, 0.000000, 0.00000, 0.0000, 0.0000, 0.0000,
										-0.37511, 0.34930, -0.19318, -0.33846, 0.76512, 0.000000, 0.00000, 0.0000, 0.0000, 0.0000,
										-0.05886, 0.13550, 0.57684, -0.31824, 0.30204, 0.672999, 0.00000, 0.0000, 0.0000, 0.0000,
										-0.05132, 0.32787, -0.04868, 0.50784, 0.33601, 0.258037, 0.67090, 0.0000, 0.0000, 0.0000,
										-0.32894, 0.47171, -0.01778, 0.52288, -0.07089, 0.213445, 0.08243, 0.5815, 0.0000, 0.0000,
										0.39669, -0.20564, 0.58014, 0.02939, -0.10623, 0.012868, -0.11365, -0.1441, 0.6464, 0.0000,
										0.02260, -0.14246, 0.69114, -0.41224, -0.07834, -0.004462, 0.15832, 0.3687, -0.2655, 0.3065
									  };
	// timestep resolution
	static double delta				= 1.0/( (double) TIMESTEPS);
	// total number of data elements for model
	static size_t N				= 2*assets*paths*(unsigned ( double (TIMESTEPS) *T));
 #ifdef PLOT
  static size_t N2				= 2*assets*paths*(unsigned ( double (TIMESTEPS) *T));
  static unsigned paths2			= 25000;
 #endif
	// For command prompt;
	int prompt;


#else

	static double r;
	static double K;
	static double T;
	static unsigned paths;
	static unsigned assets;
	static double *input_kappa, *input_theta, *input_xi, *input_rho, *input_V0, *input_S0, *input_Y0, *input_A;
	static double delta	= 1.0/((double) TIMESTEPS);
	static size_t N;
#endif

static double seconds=0.0;
// seed for Monte Carlo
static unsigned seed = 12;
// number of timesteps per simulation path
static unsigned totalTimesteps = unsigned( T*((double) TIMESTEPS));
// gamma values for Heston Price procceses
static double gamma1 = 0.5;
static double gamma2 = 0.5;
// size of random number lookup table
size_t ICDFSize = 8192;


// OpenCL Objects
static cl_platform_id platform;
#ifdef M
static cl_device_id device[MAX_DEVICES];
static cl_command_queue queuePATHGEN[MAX_DEVICES], queueRNG[MAX_DEVICES];
static cl_command_queue queueSQRT, queueEXP, queueLOG, queueUNR, queueCORRAND;
#else
static cl_device_id device;
static cl_command_queue queueSQRT, queueEXP, queueLOG, queueUNR, queueCORRAND, queuePATHGEN;
#endif
static cl_context context;
//static cl_command_queue queueSQRT, queueEXP, queueLOG, queueUNR, queueCORRAND, queuePATHGEN;
//static cl_command_queue* queuePATHGEN;
static cl_kernel kernelSQRT, kernelEXP, kernelLOG, kernelUNR, kernelCORRAND, kernelRNG[MAX_DEVICES], kernelPATHGEN[MAX_DEVICES];
static cl_program programSQRT, programEXP, programLOG, programUNR, programCORRAND, programPATHGEN;
static cl_int status;
#ifdef PLOT
static cl_event kernel_event[MAX_DEVICES],kernel_event_RNG[MAX_DEVICES];
#else
static cl_event kernel_event;
#endif
// OpenCL Memory Transfer Objects 
static cl_mem kernel_ICDF, kernel_input1, kernel_input2;
static cl_mem kernel_rnArray, kernel_unArray, kernel_kappa, kernel_invKappa, kernel_theta, kernel_xi, kernel_xi2, kernel_expKD, kernel_nexpKD,
				kernel_nexpKD2, kernel_S0, kernel_V0, kernel_rho, kernel_invsqrtRho, kernel_K0, kernel_K1, kernel_K2, kernel_K3, kernel_K4,
				kernel_retVolPaths, kernel_retPricePaths, kernel_returnValues, kernel_returnUN;
// Host pointers
static void *returnValues, *ICDF, *input1, *input2;
static void *rnArray, *unArray, *kappa, *invKappa, *theta, *xi, *xi2, *expKD, *nexpKD, *nexpKD2, *S0, *V0, 
				*rho, *invsqrtRho, *K0, *K1, *K2, *K3, *K4, *retVolPaths, *retPricePaths, *returnUN, *tempRN;
