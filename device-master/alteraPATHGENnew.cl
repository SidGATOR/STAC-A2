/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *       http://www.pcg-random.org
 */

/*
 * This code is derived from the full C implementation, which is in turn
 * derived from the canonical C++ PCG implementation. The C++ version
 * has many additional features and is preferable if you can use C++ in
 * your project.
 */

#include "pcg_basic.h"
#include "main.h"

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_altera_channels : enable
//typedef float DATATYPE

#define NUM_RNG 2
#define CLAMP_ZERO 0x1.0p-126f 
#define CLAMP_ONE  0x1.fffffep-1f

//OpenCL channels//
channel float4 RANDOM_STREAM_0 __attribute__((depth(8)));
//channel float2 UNIFORM_STREAM_0 __attribute__((depth(8)));
// state for global RNGs
//const pcg32_random_t pcg32_global = PCG32_INITIALIZER;

// pcg32_srandom(initstate, initseq)
// pcg32_srandom_r(rng, initstate, initseq):
//     Seed the rng.  Specified in two parts, state initializer and a
//     sequence selection constant (a.k.a. stream id)

void pcg32_srandom_r(pcg32_random_t* rng, ulong initstate, ulong initseq)
{
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg32_random_r(rng);
    rng->state += initstate;
    pcg32_random_r(rng);
}
// pcg32_random()
// pcg32_random_r(rng)
//     Generate a uniformly distributed 32-bit random number

DATATYPE pcg32_random_r(pcg32_random_t* rng)
{
    ulong  oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    unsigned int xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    unsigned int rot = oldstate >> 59u;
	DATATYPE temp = (float) ((xorshifted >> rot) | (xorshifted << ((-rot) & 31)))/4294967296.0f;
	//To restrict data in the range of [0,1]
	if (temp == 0.0f) temp = CLAMP_ZERO; 
    if (temp == 1.0f) temp = CLAMP_ONE;
	return temp;
}

DATATYPE RNcorr( DATATYPE zv, DATATYPE zs, DATATYPE rho, DATATYPE invsqrtRho)
{
	return (zs - zv*rho)*invsqrtRho;
}

DATATYPE varProcess(DATATYPE kappa, DATATYPE invKappa, DATATYPE theta, DATATYPE xi2, DATATYPE V0, DATATYPE expKD, DATATYPE nexpKD, DATATYPE nexpKD2, 
	DATATYPE Zv, DATATYPE Uv)
{
	DATATYPE invPsi;
	DATATYPE psic		= 1.5;
	DATATYPE m		= theta + (V0 - theta)*expKD;
	DATATYPE s2		= ( (V0*xi2*expKD)*invKappa) * nexpKD + ( (theta*xi2) * 0.5*invKappa) * nexpKD2;
	DATATYPE psi		= s2/(m*m);
	DATATYPE two_psi	= 2.0/psi;
	DATATYPE b2		= (two_psi) - 1.0 + sqrt(two_psi)*sqrt(two_psi-1.0);
	DATATYPE a		= m / (1.0+b2);
/*	
	if(psi <= psic)
	{
		return a*pow((sqrt(b2)+Zv),2);
	} else{
		DATATYPE p	= (psi-1)/(psi+1);
		DATATYPE beta	= (1-p)/m;
		if(Uv > p)
		{
			invPsi = (1/beta)*log( (1-p)/(1-Uv) );
		} else{
			invPsi = 0.0;
		}
		return invPsi;
	}
*/	 	
	DATATYPE var1 = a*pow((sqrt(b2)+Zv),2);
	DATATYPE p	= (psi-1.0)/(psi+1.0);
	DATATYPE beta	= (1.0-p)/m;
	invPsi = (1.0/beta)*log( (1.0-p)/(1.0-Uv));
	DATATYPE var2 = invPsi*(DATATYPE)(Uv>p);
	ulong retVal = ( *(ulong*) &var1 &(psi<=psic)) | ( *(ulong*) &var2&(psi>psic));
	return *(DATATYPE*) &retVal; 

}

DATATYPE priceProcess(DATATYPE theta, DATATYPE kappa, DATATYPE delta, DATATYPE rho, DATATYPE xi, DATATYPE S0, DATATYPE V0, DATATYPE V1, 
	DATATYPE Zs, DATATYPE K0, DATATYPE K1, DATATYPE K2, DATATYPE K3, DATATYPE K4)
{
	return S0+K0+K1*V0+K2*V1+Zs*sqrt(K3*V0+K4*V1);
}

float2 box_muller(float a, float b)
{
   float radius = sqrt(-2.0f * log(a));
   float angle = 2.0f*b;
   float2 result;
   result.x = radius*cospi(angle);
   result.y = radius*sinpi(angle);
   return result;
}

kernel
//__attribute((num_compute_units(2)))
//__attribute((num_simd_work_items(1)))
__attribute((reqd_work_group_size(20, ASSETS, 1)))
void random_number_generator()

{
	
    float2 Uv1,Uv2,Z1,Z2;
	float4 vectorRAND;
	pcg32_random_t rng1,rng2,rng3,rng4;
	unsigned seed = 38u + 13*get_local_id(1) + 17*get_global_id(2);
	unsigned localIndex = get_local_id(2);
	unsigned temp0 = (get_group_id(0) * get_local_size(0) *  get_local_size(1) * TIMESTEPS);
	unsigned temp1 =  (get_group_id(0) * get_local_size(0) *  get_local_size(1) * TIMESTEPS);
	unsigned temp2 = (get_local_id(0)*get_local_size(1))*TIMESTEPS;
	unsigned temp3 = get_local_id(1)*TIMESTEPS;
	unsigned retPath = temp1 + temp2 + temp3;
    pcg32_srandom_r(&rng1, seed, retPath);
  // pcg32_srandom_r(&rng2, 38u, 48u);
   // pcg32_srandom_r(&rng3, 64u, 16u);
   // pcg32_srandom_r(&rng4, 24u, 58u);
    
	for(int j=0;j < TIMESTEPS; j++){
	#pragma unroll NUM_RNG //Number of random number generator should match the number of initalizations//
     for(int i = 0; i<NUM_RNG; i++){
	 Uv1[i] =  (DATATYPE) pcg32_random_r(&rng1);
	 //retPath ++;
    // Uv2[i] = (DATATYPE) pcg32_random_r(&rng2);
      } 

     #pragma unroll NUM_RNG 
     for(int i = 0; i<NUM_RNG; i++){
     Z1 = box_muller(Uv1[i],Uv1[(i+1)%NUM_RNG]);
    // Z2 = box_muller(Uv2[i],Uv2[(i+1)%NUM_RNG]);
      }
 vectorRAND = (float4) ((float2) Z1, (float2) Uv1);
  write_channel_altera(RANDOM_STREAM_0,vectorRAND);
  //write_channel_altera(RANDOM_STREAM_0, Z1);
  //write_channel_altera(UNIFORM_STREAM_0,Uv1);
/*
	for(int i =0; i< NUM_RNG;i++){
	  retRandBuf[retPath + i%2] = Uv1[i];
	  }
//    printf("retIndex RNG: %d\n", retPath);	
	  retPath+=1;
*/
	}


}

kernel 
//__attribute((num_compute_units(2)))
//__attribute((num_simd_work_items(4)))
__attribute((reqd_work_group_size(20, ASSETS, 1)))
//__attribute((num_vector_lanes(1)))

 void alteraPATHGEN(
  DATATYPE delta,
  unsigned timesteps,
  __global const DATATYPE *restrict kappa,
  __global const DATATYPE *restrict invKappa,
  __global const DATATYPE *restrict theta,
  __global const DATATYPE *restrict xi,
  __global const DATATYPE *restrict xi2,
  __global const DATATYPE *restrict expKD,
  __global const DATATYPE *restrict nexpKD,
  __global const DATATYPE *restrict nexpKD2,
  __global const DATATYPE *restrict So,
  __global const DATATYPE *restrict Vo,
  __global const DATATYPE *restrict rho,
  __global const DATATYPE *restrict invsqrt_rho,
  __global const DATATYPE *restrict K0,
  __global const DATATYPE *restrict K1,
  __global const DATATYPE *restrict K2,
  __global const DATATYPE *restrict K3,
  __global const DATATYPE *restrict K4,
  //__global DATATYPE *restrict retVolPaths,
  __global DATATYPE *restrict retPricePaths
//  __ global DATATYPE *restrict retRandBuf
  ) {

//	local DATATYPE pathArray[ASSETS*252]; 

	unsigned localIndex		= get_local_id(1);
	DATATYPE localKappa		= kappa[localIndex];
	DATATYPE localInvKappa	= invKappa[localIndex];
	DATATYPE localTheta		= theta[localIndex];
	DATATYPE localXi			= xi[localIndex];
	DATATYPE localXi2			= xi2[localIndex];
	DATATYPE localExpKD		= expKD[localIndex];
	DATATYPE localNexpKD		= nexpKD[localIndex];
	DATATYPE localNexpKD2		= nexpKD2[localIndex];
	DATATYPE localRho			= rho[localIndex];
	DATATYPE localInvsqrt_rho	= invsqrt_rho[localIndex];
	DATATYPE localK0			= K0[localIndex];
	DATATYPE localK1			= K1[localIndex];
	DATATYPE localK2			= K2[localIndex];
	DATATYPE localK3			= K3[localIndex];
	DATATYPE localK4			= K4[localIndex];


	unsigned int VECTOR = 1<<NUM_RNG;
	DATATYPE Zv,Zs;
	DATATYPE Uv[2*NUM_RNG];
    DATATYPE Z[2*NUM_RNG];
	DATATYPE V0,V1,S0,S1;
	
	unsigned temp1 =  (get_group_id(0) * get_local_size(0) *  get_local_size(1) * TIMESTEPS);
	unsigned temp2 = (get_local_id(0)*get_local_size(1))*TIMESTEPS;
	unsigned temp3 = get_local_id(1)*TIMESTEPS;
	unsigned retPathIndex = temp1 + temp2 + temp3;
//	unsigned mySeed = randSeeds + 13*get_local_id(1) + 17*get_global_id(0);
	V0 = Vo[localIndex];
    S0 = So[localIndex];
    float4 vectorRAND;
	float2 vectorZ;
	//float2 vectorU;
	for(int t = 0; t < timesteps; t++){

	    vectorRAND = read_channel_altera(RANDOM_STREAM_0);
	    //vectorU = read_channel_altera(UNIFORM_STREAM_0);
		
		Zs = RNcorr( vectorRAND.x, vectorRAND.y, localRho, localInvsqrt_rho);
            Zv = vectorRAND.x;
        V1 = varProcess(localKappa, localInvKappa, localTheta, localXi2, V0, localExpKD, localNexpKD, localNexpKD2, Zv, vectorRAND.z);
        S1 = priceProcess(localTheta, localKappa, delta, localRho, localXi, S0, V0, V1, Zs, localK0, localK1, localK2, localK3, localK4);

	//	retVolPaths[retPathIndex]	= V1;
		retPricePaths[retPathIndex]	= exp(S1);

	retPathIndex += 1;
//	printf("retIndex RNG: %d\n",retPathIndex);
        V0 = V1;
        S0 = S1;
	}

	
}
