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
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
//typedef float DATATYPE;
typedef float16 VECTOR_DATA;
typedef float8 VECTOR_DATATYPE;

#define NUM_RNG 5
#define PROCESS 10
#define BOX_THROUGH 2
#define CLAMP_ZERO 0x1.0p-126f 
#define CLAMP_ONE  0x1.fffffep-1f

//OpenCL channels//
channel float16 CORRAND_STREAM_0 __attribute__((depth(8)));

//Test Channels//
channel float4 CORRAND_STREAM_T0 __attribute__((depth(8)));
channel float4 BLK_REDUCE_STREAM_0 __attribute__((depth(8)));
channel float4 BLK_REDUCE_STREAM_1 __attribute__((depth(8)));

channel float4 ASSEMBLE_R_STREAM_0 __attribute__((depth(8)));
channel float4 ASSEMBLE_R_STREAM_1 __attribute__((depth(8)));
channel float  ASSEMBLE_R_STREAM_2 __attribute__((depth(8)));

channel float  CASHFLOW_STREAM_0  __attribute__((depth(8)));
channel float  CASHFLOW_THROUGH_REDUCE_STREAM_0  __attribute__((depth(8)));

channel float  ASSET_STREAM_0  __attribute__((depth(8)));
channel float  ASSET_THROUGH_REDUCE_STREAM_0  __attribute__((depth(8)));

channel float  NUM_PAYPATH_STREAM_0  __attribute__((depth(8)));
channel float  NUM_PAYPATH_THROUGH_REDUCE_STREAM_0  __attribute__((depth(8)));

channel float4 BETA_STREAM_0 __attribute__((depth(8)));
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
/*
void pcg32_srandom(uint64_t seed, uint64_t seq)
{
    pcg32_srandom_r(&pcg32_global, seed, seq);
}
*/
// pcg32_random()
// pcg32_random_r(rng)
//     Generate a uniformly distributed 32-bit random number

DATATYPE maxKERNEL(DATATYPE D1, DATATYPE D2)
{
	if(D1==D2){
		return D1;
	}
	if(D1>D2){
		return D1;
	}
	else{
		return D2;
	}
}

DATATYPE absKERNEL(DATATYPE D1)
{
	if(D1 < 0.0)
	{
		return -1.0*D1;
	}
	else
	{
		return D1;
	}
}

float2 swap(DATATYPE A, DATATYPE B)
{
	DATATYPE t = A;
	A = B;
	B = t;
	float2 vectorSWAP = (float2) (A,B);
	return vectorSWAP;
}

DATATYPE off_diag_norm(DATATYPE A01, DATATYPE A02, DATATYPE A12)
{
  return sqrt(2.0 * (A01*A01 + A02*A02 + A12*A12));
}

float8 assembleR(DATATYPE m, float4 sums, float4 assets)
{
	
	printf("sums1: %f, sums2: %f, sums3: %f, sums4: %f\n", sums.x, sums.y, sums.z, sums.w);
	float8 vectorR;
	DATATYPE x0 = assets.x;
	DATATYPE x1 = assets.y;
	DATATYPE x2 = assets.z;
	
	DATATYPE x0_sq = x0 * x0;

  DATATYPE sum1 = sums.x - x0;
  DATATYPE sum2 = sums.y - x0_sq;
  DATATYPE sum3 = sums.z - x0_sq*x0;
  DATATYPE sum4 = sums.w - x0_sq*x0_sq;

  DATATYPE m_as_dbl = m;
  DATATYPE sigma = m_as_dbl - 1.0;
  DATATYPE mu = sqrt(m_as_dbl);
  DATATYPE v0 = -sigma / (1.0 + mu);
  DATATYPE v0_sq = v0*v0;
  double  beta = 2.0 * v0_sq / (sigma + v0_sq);
  
  DATATYPE inv_v0 = 1.0 / v0;
  DATATYPE one_min_beta = 1.0 - beta;
  DATATYPE beta_div_v0  = beta * inv_v0;
  
  assets = (float4) (mu, one_min_beta*x0 - beta_div_v0*sum1,one_min_beta*x0_sq - beta_div_v0*sum2,0.0);
  //printf("asset1: %f, asset2: %f, asset3: %f\n", assets.x, assets.y, assets.z);
  /*
  smem_svds[0] = mu;
  smem_svds[1] = one_min_beta*x0 - beta_div_v0*sum1;
  smem_svds[2] = one_min_beta*x0_sq - beta_div_v0*sum2;
  */
  // Rank update coefficients.
  
  DATATYPE beta_div_v0_sq = beta_div_v0 * inv_v0;
  
  DATATYPE c1 = beta_div_v0_sq*sum1 + beta_div_v0*x0;
  DATATYPE c2 = beta_div_v0_sq*sum2 + beta_div_v0*x0_sq;

  // 2nd step of QR.
  
  DATATYPE x1_sq = x1*x1;

  sum1 -= x1;
  sum2 -= x1_sq;
  sum3 -= x1_sq*x1;
  sum4 -= x1_sq*x1_sq;
  
  x0 = x1-c1;
  x0_sq = x0*x0;
  sigma = sum2 - 2.0*c1*sum1 + (m_as_dbl-2.0)*c1*c1;
  if( absKERNEL(sigma) < 1.0e-16 )
    beta = 0.0;
  else
  {
    mu = sqrt(x0_sq + sigma);
    if( x0 <= 0.0 )
      v0 = x0 - mu;
    else
      v0 = -sigma / (x0 + mu);
    v0_sq = v0*v0;
	printf("v0_sq: %f\n", v0_sq);
	printf("Beta before: %f\n",beta);
    beta = 2.0*v0_sq / (sigma + v0_sq);
	printf("Beta after: %e\n",beta);
  }
 printf("absSIGMA: %f\n",absKERNEL(sigma)); 
  inv_v0 = 1.0 / v0;
  beta_div_v0 = beta * inv_v0;
  
  // The coefficient to perform the rank update.
  DATATYPE c3 = (sum3 - c1*sum2 - c2*sum1 + (m_as_dbl-2.0)*c1*c2)*beta_div_v0;
  DATATYPE c4 = (x1_sq-c2)*beta_div_v0 + c3*inv_v0;
  DATATYPE c5 = c1*c4 - c2;
  
  one_min_beta = 1.0 - beta;
  
  assets.w = one_min_beta*x0 - beta_div_v0*sigma;
  float2 updateR; 
  updateR.x = one_min_beta*(x1_sq-c2) - c3;
  /*
  // Update R. 
  smem_svds[3] = one_min_beta*x0 - beta_div_v0*sigma;
  smem_svds[4] = one_min_beta*(x1_sq-c2) - c3;
  */
  // 3rd step of QR.
  
  DATATYPE x2_sq = x2*x2;

  sum1 -= x2;
  sum2 -= x2_sq;
  sum3 -= x2_sq*x2;
  sum4 -= x2_sq*x2_sq;
  
  x0 = x2_sq-c4*x2+c5;
  sigma = sum4 - 2.0*c4*sum3 + (c4*c4 + 2.0*c5)*sum2 - 2.0*c4*c5*sum1 + (m_as_dbl-3.0)*c5*c5;
  if( absKERNEL(sigma) < 1.0e-12 )
    beta = 0.0;
  else
  {
    mu = sqrt(x0*x0 + sigma);
    if( x0 <= 0.0 )
      v0 = x0 - mu;
    else
      v0 = -sigma / (x0 + mu);
    v0_sq = v0*v0;
    beta = 2.0*v0_sq / (sigma + v0_sq);
  }
  updateR.y = (1.0-beta)*x0 - (beta/v0)*sigma;
  vectorR = (float8) ((float4) assets, (float2) updateR, 0.0, 0.0);
  printf("asset1: %f, asset2: %f, asset3: %f, asset4: %f, updateR1: %f, updateR2: %f\n", assets.x, assets.y, assets.z,assets.w, updateR.x, updateR.y);
  return vectorR;
  /*
  write_channel_altera(ASSEMBLE_R_STREAM_0, vectorR);
  // Update R.
  smem_svds[5] = (1.0-beta)*x0 - (beta/v0)*sigma;
  */
}

float16 svd3x3(DATATYPE m, float4 sums, float8 vectorR)
{
	// The matrix R.
  DATATYPE R00 = vectorR.s0;
  DATATYPE R01 = vectorR.s1;
  DATATYPE R02 = vectorR.s2;
  DATATYPE R11 = vectorR.s3;
  DATATYPE R12 = vectorR.s4;
  DATATYPE R22 = vectorR.s5;
  
  // We compute the eigenvalues/eigenvectors of A = R^T R.
  
  DATATYPE A00 = R00*R00;
  DATATYPE A01 = R00*R01;
  DATATYPE A02 = R00*R02;
  DATATYPE A11 = R01*R01 + R11*R11;
  DATATYPE A12 = R01*R02 + R11*R12;
  DATATYPE A22 = R02*R02 + R12*R12 + R22*R22;
  
   // We keep track of V since A = Sigma^2 V. Each thread stores a row of V.
  
  DATATYPE V00 = 1.0, V01 = 0.0, V02 = 0.0;
  DATATYPE V10 = 0.0, V11 = 1.0, V12 = 0.0;
  DATATYPE V20 = 0.0, V21 = 0.0, V22 = 1.0;
  
  // The Jacobi algorithm is iterative. We fix the max number of iter and the minimum tolerance.
  
  unsigned max_iters = 16;
  double tolerance = 1.0e-12;
  
  // Iterate until we reach the max number of iters or the tolerance.
 
  for( int iter = 0 ; off_diag_norm(A01, A02, A12) >= tolerance && iter < max_iters ; ++iter )
  {
    DATATYPE c, s, B00, B01, B02, B10, B11, B12, B20, B21, B22;
    
    // Compute the Jacobi matrix for p=0 and q=1.
    
    c = 1.0, s = 0.0;
    if( A01 != 0.0 )
    {
      DATATYPE tau = (A11 - A00) / (2.0 * A01);
      DATATYPE sgn = tau < 0.0 ? -1.0 : 1.0;
      DATATYPE t   = sgn / (sgn*tau + sqrt(1.0 + tau*tau));
      
      c = 1.0 / sqrt(1.0 + t*t);
      s = t*c;
    }
	
	// Update A = J^T A J and V = V J.
    
    B00 = c*A00 - s*A01;
    B01 = s*A00 + c*A01;
    B10 = c*A01 - s*A11;
    B11 = s*A01 + c*A11;
    B02 = A02;
    
    A00 = c*B00 - s*B10;
    A01 = c*B01 - s*B11;
    A11 = s*B01 + c*B11;
    A02 = c*B02 - s*A12;
    A12 = s*B02 + c*A12;
    
    B00 = c*V00 - s*V01;
    V01 = s*V00 + c*V01;
    V00 = B00;
    
    B10 = c*V10 - s*V11;
    V11 = s*V10 + c*V11;
    V10 = B10;
    
    B20 = c*V20 - s*V21;
    V21 = s*V20 + c*V21;
    V20 = B20;
	
	// Compute the Jacobi matrix for p=0 and q=2.
    
	c = 1.0, s = 0.0;
    if( A02 != 0.0 )
    {
      DATATYPE tau = (A22 - A00) / (2.0 * A02);
      DATATYPE sgn = tau < 0.0 ? -1.0 : 1.0;
      DATATYPE t   = sgn / (sgn*tau + sqrt(1.0 + tau*tau));
      
      c = 1.0 / sqrt(1.0 + t*t);
      s = t*c;
    }
	
	// Update A = J^T A J and V = V J.
    
    B00 = c*A00 - s*A02;
    B01 = c*A01 - s*A12;
    B02 = s*A00 + c*A02;
    B20 = c*A02 - s*A22;
    B22 = s*A02 + c*A22;
    
    A00 = c*B00 - s*B20;
    A12 = s*A01 + c*A12;
    A02 = c*B02 - s*B22;
    A22 = s*B02 + c*B22;
    A01 = B01;
    
    B00 = c*V00 - s*V02;
    V02 = s*V00 + c*V02;
    V00 = B00;
    
    B10 = c*V10 - s*V12;
    V12 = s*V10 + c*V12;
    V10 = B10;
    
    B20 = c*V20 - s*V22;
    V22 = s*V20 + c*V22;
    V20 = B20;
	
	// Compute the Jacobi matrix for p=1 and q=2.
    
    c = 1.0, s = 0.0;
    if( A12 != 0.0 )
    {
      DATATYPE tau = (A22 - A11) / (2.0 * A12);
      DATATYPE sgn = tau < 0.0 ? -1.0 : 1.0;
      DATATYPE t   = sgn / (sgn*tau + sqrt(1.0 + tau*tau));
      
      c = 1.0 / sqrt(1.0 + t*t);
      s = t*c;
    }
	
	// Update A = J^T A J and V = V J.
    
    B02 = s*A01 + c*A02;
    B11 = c*A11 - s*A12;
    B12 = s*A11 + c*A12;
    B21 = c*A12 - s*A22;
    B22 = s*A12 + c*A22;
    
    A01 = c*A01 - s*A02;
    A02 = B02;
    A11 = c*B11 - s*B21;
    A12 = c*B12 - s*B22;
    A22 = s*B12 + c*B22;
    
    B01 = c*V01 - s*V02;
    V02 = s*V01 + c*V02;
    V01 = B01;
    
    B11 = c*V11 - s*V12;
    V12 = s*V11 + c*V12;
    V11 = B11;
    
    B21 = c*V21 - s*V22;
    V22 = s*V21 + c*V22;
    V21 = B21;
  }
  
  float2 vectorSWAP1,vectorSWAP2,vectorSWAP3, vectorSWAP4;
  // Swap the columns to have S[0] >= S[1] >= S[2].
  if( A00 < A11 )
  {
    vectorSWAP1 = swap(A00, A11); A00 = vectorSWAP1.x; A11 = vectorSWAP1.y;
    vectorSWAP2 = swap(V00, V01); V00 = vectorSWAP2.x; V01 = vectorSWAP2.y; 
    vectorSWAP3 = swap(V10, V11); V10 = vectorSWAP3.x; V11 = vectorSWAP3.y;
    vectorSWAP4 = swap(V20, V21); V20 = vectorSWAP4.x; V21 = vectorSWAP4.y;
  }
  if( A00 < A22 )
  {
    vectorSWAP1 = swap(A00, A22); A00 = vectorSWAP1.x; A22 = vectorSWAP1.y;
    vectorSWAP2 = swap(V00, V02); V00 = vectorSWAP2.x; V02 = vectorSWAP2.y; 
    vectorSWAP3 = swap(V10, V12); V10 = vectorSWAP3.x; V12 = vectorSWAP3.y;
    vectorSWAP4 = swap(V20, V22); V20 = vectorSWAP4.x; V22 = vectorSWAP4.y;
	/*
    swap(A00, A22);
    swap(V00, V02);
    swap(V10, V12);
    swap(V20, V22);
	*/
  }
  if( A11 < A22 )
  {
    vectorSWAP1 = swap(A11, A22); A11 = vectorSWAP1.x; A22 = vectorSWAP1.y;
    vectorSWAP2 = swap(V01, V02); V01 = vectorSWAP2.x; V02 = vectorSWAP2.y; 
    vectorSWAP3 = swap(V11, V12); V11 = vectorSWAP3.x; V12 = vectorSWAP3.y;
    vectorSWAP4 = swap(V21, V22); V21 = vectorSWAP4.x; V22 = vectorSWAP4.y;
	/*
    swap(A11, A22);
    swap(V01, V02);
    swap(V11, V12);
    swap(V21, V22);
	*/
  }
  
  //printf("timestep=%3d, svd0=%.8lf svd1=%.8lf svd2=%.8lf\n", get_global_id(0), sqrt(A00), sqrt(A11), sqrt(A22));
  
  // Invert the diagonal terms and compute V*S^-1.
  
  DATATYPE inv_S0 = absKERNEL(A00) < 1.0e-12 ? 0.0 : 1.0 / A00;
  DATATYPE inv_S1 = absKERNEL(A11) < 1.0e-12 ? 0.0 : 1.0 / A11;
  DATATYPE inv_S2 = absKERNEL(A22) < 1.0e-12 ? 0.0 : 1.0 / A22;

  // printf("SVD: timestep=%3d %12.8lf %12.8lf %12.8lf\n", get_global_id(0), sqrt(A00), sqrt(A11), sqrt(A22));
  
  DATATYPE U00 = V00 * inv_S0; 
  DATATYPE U01 = V01 * inv_S1; 
  DATATYPE U02 = V02 * inv_S2;
  DATATYPE U10 = V10 * inv_S0; 
  DATATYPE U11 = V11 * inv_S1; 
  DATATYPE U12 = V12 * inv_S2;
  DATATYPE U20 = V20 * inv_S0; 
  DATATYPE U21 = V21 * inv_S1; 
  DATATYPE U22 = V22 * inv_S2;
  
  // Compute V*S^-1*V^T*R^T.
  
  DATATYPE B00 = U00*V00 + U01*V01 + U02*V02;
  DATATYPE B01 = U00*V10 + U01*V11 + U02*V12;
  DATATYPE B02 = U00*V20 + U01*V21 + U02*V22;
  DATATYPE B10 = U10*V00 + U11*V01 + U12*V02;
  DATATYPE B11 = U10*V10 + U11*V11 + U12*V12;
  DATATYPE B12 = U10*V20 + U11*V21 + U12*V22;
  DATATYPE B20 = U20*V00 + U21*V01 + U22*V02;
  DATATYPE B21 = U20*V10 + U21*V11 + U22*V12;
  DATATYPE B22 = U20*V20 + U21*V21 + U22*V22;
  
  vectorR.s6 = B00*R00 + B01*R01 + B02*R02;
  vectorR.s7 = B01*R11 + B02*R12;
  float16 vectorSVD;
  vectorSVD = (float16) ((float8) vectorR, B02*R22, B10*R00 + B11*R01 + B12*R02, B11*R11 + B12*R12, B12*R22, B20*R00 + B21*R01 + B22*R02, B21*R11 + B22*R12, B22*R22, 0.0);
  return vectorSVD;
  /*
  smem_svds[ 6] = B00*R00 + B01*R01 + B02*R02;
  smem_svds[ 7] =           B01*R11 + B02*R12;
  smem_svds[ 8] =                     B02*R22;
  smem_svds[ 9] = B10*R00 + B11*R01 + B12*R02;
  smem_svds[10] =           B11*R11 + B12*R12;
  smem_svds[11] =                     B12*R22;
  smem_svds[12] = B20*R00 + B21*R01 + B22*R02;
  smem_svds[13] =           B21*R11 + B22*R12;
  smem_svds[14] =                     B22*R22;
  */
}


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

float2 box_muller(DATATYPE a, DATATYPE b)
{
   DATATYPE radius = sqrt(-2.0f * log(a));
   DATATYPE angle = 2.0f*b;
   float2 result;
   result.x = radius*cospi(angle);
   result.y = radius*sinpi(angle);
   return result;
}

kernel
//__attribute((num_compute_units(2)))
//__attribute((num_simd_work_items(1)))
__attribute((reqd_work_group_size(20, 1, 1)))
void random_number_generator()

{
	
	DATATYPE corrMATRIX[10][10] = {
	{1.00000000, 0.25887629, -0.01500694, 0.14624317, 0.09627141, 0.02864714, -0.13019061, 0.27598592, -0.12258651,-0.07830565},
	{0.25887629, 1.00000000,  0.01283857, 0.44538116, 0.70229933, 0.25435678,0.27730185,  0.42426839,  0.05412770, -0.40095947},
    {-0.01500694, 0.01283857, 1.00000000, 0.24030844, -0.43379920, -0.42379917, -0.10070932, -0.23880511,  0.13239119 -0.30242927},
    {0.14624317, 0.44538116, 0.24030844,  1.00000000,  0.23862474, -0.06606606, 0.19779575,  0.42422825, -0.03171912, -0.41791624},
    {0.09627141, 0.70229933, -0.43379920,  0.23862474,  1.00000000,  0.56199770, -0.15985347,  0.22418479,  0.05150370, -0.10808895},
    {0.02864714, 0.25435678, -0.42379917, -0.06606606,  0.56199770,  1.00000000, -0.08033369,  0.02434509,  0.09797083,  0.69902731},
    {-0.13019061, 0.27730185, -0.10070932,  0.19779575, -0.15985347, -0.08033369, 1.00000000,  0.30084694, -0.31723827, -0.13884371},
    {0.27598592, 0.42426839, -0.23880511,  0.42422825,  0.22418479,  0.02434509, 0.30084694,  1.00000000,  0.06219817, -0.17674700},
    {-0.12258651, 0.05412770, 0.13239119, -0.03171912,  0.05150370,  0.09797083, -0.31723827,  0.06219817,  1.00000000,  0.02719567},
    {-0.07830565, -0.40095947, -0.30242927, -0.41791624, -0.10808895,  0.69902731,-0.13884371, -0.17674700,  0.02719567,  1.00000000}};
    
	DATATYPE corrRAND[10],RAND[16];
	float16 vectorCORRAND;
	float4 vectorCORRANDx4;
	float2 z1,z2,z3,z4,z5;
	
	DATATYPE u[10],z[10]; 
	float16 vectorRAND;
	pcg32_random_t rng1,rng2,rng3,rng4,rng5;
	unsigned seed = 38u + 13*get_local_id(1) + 17*get_global_id(2);
	unsigned localIndex = get_local_id(2);
	unsigned temp0 = (get_group_id(0) * get_local_size(0) *  get_local_size(1) * TIMESTEPS);
	unsigned temp1 =  (get_group_id(0) * get_local_size(0) *  get_local_size(1) * TIMESTEPS);
	unsigned temp2 = (get_local_id(0)*get_local_size(1))*TIMESTEPS;
	unsigned temp3 = get_local_id(1)*TIMESTEPS;
	unsigned retPath = temp1 + temp2 + temp3;
    pcg32_srandom_r(&rng1, seed, retPath);
    pcg32_srandom_r(&rng2, seed + 14u, retPath);
    pcg32_srandom_r(&rng3, seed + 73u, retPath);
    pcg32_srandom_r(&rng4, seed + 49u, retPath);
    pcg32_srandom_r(&rng5, seed + 64u, retPath);

  // pcg32_srandom_r(&rng2, 38u, 48u);
   // pcg32_srandom_r(&rng3, 64u, 16u);
   // pcg32_srandom_r(&rng4, 24u, 58u);
    
	for(int j=0;j < TIMESTEPS; j++){
	#pragma unroll BOX_THROUGH //Number of random number generator should match the number of initalizations//
     for(int i = 0; i<BOX_THROUGH; i++){
	 u[i] =  (DATATYPE) pcg32_random_r(&rng1);
	 u[i+2] =  (DATATYPE) pcg32_random_r(&rng2);
	 u[i+4] =  (DATATYPE) pcg32_random_r(&rng3);
	 u[i+6] =  (DATATYPE) pcg32_random_r(&rng4);
	 u[i+8] =  (DATATYPE) pcg32_random_r(&rng5);
	 //retPath ++;
    // Uv2[i] = (DATATYPE) pcg32_random_r(&rng2);
      }

	DATATYPE tempZ[PROCESS];
	/*
	z1 = box_muller(u[0],u[1]);
	z2 = box_muller(u[2],u[3]);
	z3 = box_muller(u[4],u[5]);
	z4 = box_muller(u[6],u[7]);
	z5 = box_muller(u[8],u[9]);
	*/
#pragma unroll ASSETS
	for(int i=0;i<ASSETS;i++){
	z1 = box_muller(u[2*i],u[2*i+1]);
	tempZ[2*i] = z1.x;
	tempZ[2*i+1] = z1.y;
	}

		
//vectorRAND = (float16) ((float2) z1,(float2) z2,(float2) z3,(float2) z4,(float2) z5,u[0],u[1],u[2],u[3],u[4],u[5]);
	/*
#pragma unroll 16
	for(int i=0; i< 16;i++){
		RAND[i] = vectorRAND[i];
	}
*/
	//initializing corrRAND
#pragma unroll PROCESS
	for(int i=0;i<10;i++){
	corrRAND[i] = 0.0f;
	}
	//MATRIX CORRELATION MULTIPLICATION
	 for(int x = 0; x<1; x++){
		for(int y=0;y<10;y++) {
			for(int z=0;z<10;z++){
            corrRAND[x*10+y] += tempZ[x*10+z] * corrMATRIX[z][y];
			}
		}
	}

//	vectorCORRAND = (float16) (corrRAND[0], corrRAND[1], corrRAND[2], corrRAND[3], corrRAND[4], corrRAND[5], corrRAND[6], corrRAND[7], corrRAND[8], corrRAND[9],RAND[10],RAND[11],RAND[12],RAND[13],RAND[14],RAND[15]);
	
	for(int i=0;i<ASSETS;i++){
	vectorCORRANDx4 = (float4) (corrRAND[(2*i)%10],corrRAND[(2*i+1)%10],u[i+10],u[i+11]);
    write_channel_altera(CORRAND_STREAM_T0,vectorCORRANDx4);
	}
   // write_channel_altera(CORRAND_STREAM_0,vectorCORRAND);
	}
//	printf("PATH ID: %d\n",get_global_id(0));
}


kernel 
//__attribute((num_compute_units(2)))
//__attribute((num_simd_work_items(4)))
__attribute((reqd_work_group_size(20, 1, 1)))
//__attribute((num_vector_lanes(1)))

 void alteraPATHGEN(
  float delta,
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
  __global DATATYPE *restrict retPricePaths,
  unsigned paths,
  float K
//  __ global DATATYPE *restrict retRandBuf
  ) {

//	local DATATYPE pathArray[ASSETS*252]; 

	DATATYPE localKappa;
	DATATYPE localInvKappa;
	DATATYPE localTheta;
	DATATYPE localXi;
	DATATYPE localXi2;
	DATATYPE localExpKD;
	DATATYPE localNexpKD;
	DATATYPE localNexpKD2;
	DATATYPE localRho;
	DATATYPE localInvsqrt_rho;
	DATATYPE localK0;
	DATATYPE localK1;
	DATATYPE localK2;
	DATATYPE localK3;
	DATATYPE localK4;
	DATATYPE V0;
    DATATYPE S0;
	DATATYPE Zv[ASSETS],Zs[ASSETS];
	DATATYPE Uv;
    DATATYPE Z;
	DATATYPE V1[ASSETS],S1[ASSETS];
	DATATYPE assetMAX=0.0f;
    DATATYPE tempMAX;
	unsigned localPATHS = paths;
	DATATYPE localK = K;
//	printf("Delta :\t%f\n",delta);
	
for(int t = 0; t < timesteps; t++){
    for(int i=0;i<ASSETS; i++){
	
	 localKappa		= kappa[i];
	 localInvKappa	= invKappa[i];
	 localTheta		= theta[i];
	 localXi		= xi[i];
	 localXi2			= xi2[i];
	 localExpKD	= expKD[i];
	 localNexpKD		= nexpKD[i];
	 localNexpKD2		= nexpKD2[i];
	 localRho			= rho[i];
	 localInvsqrt_rho	= invsqrt_rho[i];
	 localK0			= K0[i];
	 localK1			= K1[i];
	 localK2			= K2[i];
	 localK3			= K3[i];
	 localK4			= K4[i];
	 V0 = Vo[i];
     S0 = So[i];


	unsigned int VECTOR = 1<<NUM_RNG;

	unsigned temp1 =  (get_group_id(0) * get_local_size(0) * ASSETS  * TIMESTEPS);
	unsigned temp2 = (get_local_id(0) * ASSETS * TIMESTEPS);
	unsigned temp3 = 0; //get_local_id(1)*TIMESTEPS;
	unsigned retPathIndex = temp1 + temp2 + temp3;
    float4 vectorZ;
	float8 vectorASSET;

	    vectorZ = read_channel_altera(CORRAND_STREAM_T0);
		Zs[i] = RNcorr( vectorZ[i%2], vectorZ[(i+1)%2], localRho, localInvsqrt_rho);
            Zv[i] = vectorZ[i%2];
        V1[i] = varProcess(localKappa, localInvKappa, localTheta, localXi2, V0, localExpKD, localNexpKD, localNexpKD2, Zv[i], vectorZ[3]);
        S1[i] = priceProcess(localTheta, localKappa, delta, localRho, localXi, S0, V0, V1[i], Zs[i], localK0, localK1, localK2, localK3, localK4);
	//	retVolPaths[retPathIndex]	= V1;
//	printf("retIndex RNG: %d\n",retPathIndex);
        V0 = V1[i];
        S0 = S1[i];
//		vectorASSET = (float8) (S1[0],S1[1],S1[2],S1[3],S1[4],localK,localPATHS,0.0f);
//		write_channel_altera(ASSET_STREAM_0,vectorASSET);
	
		//retPricePaths[retPathIndex] = exp(S1[i]);
		//retPricePaths++;

assetMAX = maxKERNEL(assetMAX,maxKERNEL(exp(S1[i]),0.0f));
}
	retPricePaths[t*localPATHS+get_local_id(0)+get_group_id(0)*get_local_size(0)] = maxKERNEL( maxKERNEL(assetMAX,localK), 0.0);


}
}

kernel
__attribute((reqd_work_group_size(1, 1, 1)))
void pseudo_inverse(
  float K,
  unsigned path,
  __global DATATYPE *restrict assetMAX)

{
	float4 Sums;
	float4 vectorBlockReduce;
	float4 vectorAssetMax;
	DATATYPE localK = K;
	unsigned localPATH = path;
	unsigned cashflowID = 0;
	//DATATYPE assetMAX;
	DATATYPE localMAX;
	unsigned m = 0;
	unsigned localID;
//	printf("Strike Price inside kernel: %f\n",K);

	for(int i =0;i<localPATH;i++)
		{
			localID = get_global_id(0);
		//	localID = get_local_id(0) + get_group_id(0)*get_local_size(0);
			//assetMAX  = retPricePaths[i+(TIMESTEPS - localID)];
	//		printf("assetMAX: %f\n",assetMAX[i+localID*localPATH]);
			localMAX = assetMAX[i+localID*localPATH];
			DATATYPE x = 0.0, x_sq = 0.0;
			if(localMAX > localK){
				m = 1;
				x = localMAX;
				x_sq = localMAX*localMAX;
				//Sums.x += localK;
			}
		//	printf("index #: %d,m per path: %d\n",i,m);	
				Sums.x = x;
				Sums.y = x_sq;
				Sums.z = x_sq*x;
				Sums.w = x_sq*x_sq;
				vectorBlockReduce = (float4) ( (float4) Sums);
			//	vectorAssetMax = (float4) (assetMAX[0], assetMAX[1], assetMAX[2], m);
			//	printf("Global ID: %d Item value: %f, Sums.x: %f, m: %d\n",i +localID*path, localMAX,Sums.x,m);
				
				cashflowID = i;
				write_channel_altera(NUM_PAYPATH_STREAM_0,m);
				write_channel_altera(CASHFLOW_STREAM_0,assetMAX[cashflowID]); 
                write_channel_altera(ASSET_STREAM_0,localMAX);  
				write_channel_altera(BLK_REDUCE_STREAM_0,vectorBlockReduce);
 	     }

	vectorAssetMax = (float4)( assetMAX[localID*localPATH],assetMAX[localID*localPATH + 1], assetMAX[localID*localPATH + 2],0.0f);
	write_channel_altera(BLK_REDUCE_STREAM_1,vectorAssetMax);
}

kernel
void block_reduce(
	unsigned path)
{
	float vectorAsset, cashflow;
	float mAcc, m;
	float4 vectorReduceAcc,vectorReduce, vectorAssetMax;
	DATATYPE total_sum = 0.0;
	for(int i =0; i< TIMESTEPS;i ++)
			{
				total_sum = 0.0; mAcc = 0.0; vectorAssetMax = (float4) (0.0,0.0,0.0,0.0);
				//vectorAssetMax = read_channel_altera(BLK_REDUCE_STREAM_1);
				for(int j = 0; j<path;j++) 
				{
					//printf("Reading Started\n");
					m = read_channel_altera(NUM_PAYPATH_STREAM_0);
					cashflow = read_channel_altera(CASHFLOW_STREAM_0);
					vectorAsset = read_channel_altera(ASSET_STREAM_0);
	            	vectorReduce = read_channel_altera(BLK_REDUCE_STREAM_0);
			    	//printf("Reading Finished\n");
					vectorReduceAcc.x += vectorReduce.x;
					vectorReduceAcc.y += vectorReduce.y;	
				    vectorReduceAcc.z += vectorReduce.z; 
					vectorReduceAcc.w += vectorReduce.w;
					mAcc += m;
					//write_channel_altera(CASHFLOW_THROUGH_REDUCE_STREAM_0,cashflow);
					//write_channel_altera(ASSET_THROUGH_REDUCE_STREAM_0,vectorAsset);
					//printf("TimeStamp #: %d, path_id: %d\n",i,j);
//					printf("assetMAX[1]: %f, assetMAX[2]: %f assetMAX[3]: %f\n",vectorReduce.s5,vectorReduce.s6,vectorReduce.s7);
				}
				
				vectorAssetMax = read_channel_altera(BLK_REDUCE_STREAM_1);
				write_channel_altera(ASSEMBLE_R_STREAM_0,vectorReduceAcc);
				write_channel_altera(ASSEMBLE_R_STREAM_1,vectorAssetMax);
				write_channel_altera(ASSEMBLE_R_STREAM_2,mAcc);
//					printf("TimeStamp #: %d, m: %f\n",i,mAcc);
			}

}



kernel

void compute_beta(
	float K,
	unsigned path,
	__global DATATYPE *restrict assetMAX)
{
	unsigned localPATH = path;
	float localK = K;
	float4 sums, assets, vectorBETA;
	float8 vectorASSEMBLE_R;
	float16 vectorSVD;
	float m, asset;
	DATATYPE localM;
	unsigned cashflowID = get_global_id(0)*path;
	unsigned localID;
	//printf("Timestep: %lu, Reading Started\n",get_global_id(0));
	sums = read_channel_altera(ASSEMBLE_R_STREAM_0); 
	assets = (float4)( assetMAX[cashflowID],assetMAX[cashflowID + 1], assetMAX[cashflowID + 2],0.0f);
	printf("asset1: %f, asset2: %f, asset3: %f\n", assets.x, assets.y, assets.z);
	//assets= read_channel_altera(ASSEMBLE_R_STREAM_1);
	m = read_channel_altera(ASSEMBLE_R_STREAM_2);
	printf("m: %f\n",m);
	vectorASSEMBLE_R = assembleR(m, sums, assets);
    vectorSVD = svd3x3(m, sums, vectorASSEMBLE_R);
	
	//Loading values of R
	DATATYPE R00 = vectorSVD.s0;
	DATATYPE R01 = vectorSVD.s1;
	DATATYPE R02 = vectorSVD.s2;
	DATATYPE R11 = vectorSVD.s3;
	DATATYPE R12 = vectorSVD.s4;
	DATATYPE R22 = vectorSVD.s5;
	
	//Loding Values of W
	DATATYPE W00 = vectorSVD.s6;
	DATATYPE W01 = vectorSVD.s7;
	DATATYPE W02 = vectorSVD.s8;
	DATATYPE W10 = vectorSVD.s9;
	DATATYPE W11 = vectorSVD.sa;
	DATATYPE W12 = vectorSVD.sb;
	DATATYPE W20 = vectorSVD.sc;
	DATATYPE W21 = vectorSVD.sd;
	DATATYPE W22 = vectorSVD.se;
	
	// Invert the diagonal of R
    DATATYPE inv_R00 = R00 != 0.0 ? half_recip(R00): 0.0;
    DATATYPE inv_R11 = R11 != 0.0 ? half_recip(R11): 0.0;
    DATATYPE inv_R22 = R22 != 0.0 ? half_recip(R22): 0.0;

	//Precompute the R terms
	DATATYPE inv_R01 = inv_R00*inv_R11*R01;
	DATATYPE inv_R02 = inv_R00*inv_R22*R02;
	DATATYPE inv_R12 = inv_R22*R12;

	//Precompute W00/R00
	DATATYPE inv_W00 = W00*inv_R00;
	DATATYPE inv_W10 = W10*inv_R00;
	DATATYPE inv_W20 = W20*inv_R00;

	// Each thread has 3 numbers to sum
	DATATYPE beta0 = 0.0, beta1 = 0.0, beta2 = 0.0;
	
	for(int i = 0; i<localPATH; i++)
	{
		localID = i + get_global_id(0)*localPATH;
		asset = assetMAX[localID];
		unsigned in_the_money = 0.0;
		if(asset>localK)
		{
			in_the_money = asset-localK;
		}	
			//Computing Qis. The elements of the Q matrix in the QR decomposition
			DATATYPE Q1i = inv_R11*asset - inv_R01;
			DATATYPE Q2i = inv_R22*asset*asset - inv_R02 - Q1i*inv_R12;
			
			const DATATYPE WI0 = inv_W00 + W01 * Q1i + W02 * Q2i;
			const DATATYPE WI1 = inv_W10 + W11 * Q1i + W12 * Q2i;
			const DATATYPE WI2 = inv_W20 + W21 * Q1i + W22 * Q2i;

			DATATYPE cashflow = in_the_money ? assetMAX[cashflowID]:0.0;
		
		beta0 +=WI0*cashflow;
		beta1 +=WI1*cashflow;
		beta2 +=WI2*cashflow;
	}
	
	vectorBETA = (float4) (beta0, beta1, beta2, 0.0);
	printf("Timestep: %lu, beta0: %f, beta1: %f, beta2: %f\n",get_global_id(0),beta0, vectorBETA.y, vectorBETA.z);
	//write_channel_altera(BETA_STREAM_0,vectorBETA);
	
}


/*
kernel
void compute_final_beta(
	unsigned path)
{
 float4 vectorBETA, vectorBETAacc;


}
*/
