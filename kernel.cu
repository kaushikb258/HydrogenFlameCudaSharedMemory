#include <iostream>
#include <math.h>
#include <stdlib.h>
#include "classes.h"
#include "kernel.h"
#include "consts.h"

const int TPB = 16; // threads per block
const int RAD = 2; // number of ghost cells

//------------------------------------------------ 
// thermodynamics functions

// function compute_p
__host__ __device__ double compute_p(double rho1, double e1, double yh2, double yo2, double yh2o)
{
   double cvmix, mw, T, p;
   cvmix = yh2*(CpH2-Runiv) + yo2*(CpO2-Runiv) + yh2o*(CpH2O-Runiv);
   mw = 1.0/(yh2/MWH2 + yo2/MWO2 + yh2o/MWH2O);
   T = e1/(1000.0*cvmix/mw);
   p = rho1*(1000.0*Runiv/mw)*T; 
   return p;
}


// function compute_T
__host__ __device__ double compute_T(double rho1, double p1, double yh2, double yo2, double yh2o)
{
   double mw, T;
   mw = 1.0/(yh2/MWH2 + yo2/MWO2 + yh2o/MWH2O);
   T = p1/rho1/(1000.0*Runiv/mw);
   return T;
} 


// function compute_e
__host__ __device__ double compute_e(double rho1, double p1, double yh2, double yo2, double yh2o)
{
   double cvmix, mw, T;
   cvmix = yh2*(CpH2-Runiv) + yo2*(CpO2-Runiv) + yh2o*(CpH2O-Runiv);
   mw = 1.0/(yh2/MWH2 + yo2/MWO2 + yh2o/MWH2O);
   T = p1/rho1/(1000.0*Runiv/mw);
   return (1000.0*cvmix/mw*T);
}


// function compute_sound
__host__ __device__ double compute_sound(double rho1, double p1, double yh2, double yo2, double yh2o)
  {
   double cpmix, cvmix, gammix;
   cpmix = yh2*CpH2 + yo2*CpO2 + yh2o*CpH2O;
   cvmix = yh2*(CpH2-Runiv) + yo2*(CpO2-Runiv) + yh2o*(CpH2O-Runiv);
   gammix = cpmix/cvmix;
   return sqrt(gammix*p1/rho1);
}


//------------------------------------------------ 
// advanceKernel function

__global__ void advanceKernel(fluid *d_fl, conservative *d_Q)
{
 
 // global indices
 int col, row, id;

 // local indices
 int s_col, s_row, s_id;

 // shared memory
 const int s_size = (TPB+2*RAD)*(TPB+2*RAD);

 __shared__ double s_rho[s_size];
 __shared__ double s_u[s_size];
 __shared__ double s_v[s_size];
 __shared__ double s_p[s_size];
 __shared__ double s_e[s_size];
 __shared__ double s_YH2[s_size];
 __shared__ double s_YO2[s_size];
 __shared__ double s_YH2O[s_size];
 
 int si, di; 
 double dfdx[neqns], dfdy[neqns];

 // x-size of shared memory
 const int s_x = blockDim.x + 2*RAD;
 
 double dQ[neqns];
 double new_q[neqns];
 int l;
 double new_rho, new_u, new_v, new_e, new_p, new_yh2, new_yo2, new_yh2o, ke;
 double T, RR, concH2, concO2, k_react;
 double source[4];
 double sumYk; 
 double wavel, waver, fluxl, fluxr, fm1, f, fp1, wm1, w, wp1, consm1, cons, consp1;

//--------------------------------------
 
 // global indices
 col = threadIdx.x + blockDim.x * blockIdx.x;
 row = threadIdx.y + blockDim.y * blockIdx.y;
 if ((col >= Nx) || (row >= Ny)) return;
 id = col + row*blockDim.x*gridDim.x;

 // local indices
 s_col = threadIdx.x + RAD;
 s_row = threadIdx.y + RAD;
 s_id = s_col + s_row*s_x;


 __syncthreads(); 

//--------------------------------

 // Load regular cells  
 s_rho[s_id] = d_fl->getrho(id);
 s_u[s_id] = d_fl->getu(id);
 s_v[s_id] = d_fl->getv(id);
 s_p[s_id] = d_fl->getp(id);
 s_e[s_id] = d_fl->gete(id); 
 s_YH2[s_id] = d_fl->getYH2(id);
 s_YO2[s_id] = d_fl->getYO2(id);
 s_YH2O[s_id] = d_fl->getYH2O(id);

 // for consistency 
 s_e[s_id] = compute_e(s_rho[s_id],s_p[s_id],s_YH2[s_id],s_YO2[s_id],s_YH2O[s_id]); 


//--------------------------------


 // Load ghost cells

 // left: free slip wall
 // right: supersonic outflow
 // top: free slip wall
 // bottom: free slip wall  

//--------------------------------


 // left 
 if(blockIdx.x > 0)
 {
  // interior
  if(threadIdx.x < RAD)
  { 
   si = s_id - RAD;
   di = id - RAD;
           s_rho[si] = d_fl->getrho(di);
	   s_u[si] = d_fl->getu(di);
	   s_v[si] = d_fl->getv(di);
	   s_p[si] = d_fl->getp(di);
	   s_e[si] = d_fl->gete(di);
	   s_YH2[si] = d_fl->getYH2(di); 
	   s_YO2[si] = d_fl->getYO2(di); 
	   s_YH2O[si] = d_fl->getYH2O(di); 
	  
	   // for consistency 
	   s_e[si] = compute_e(s_rho[si],s_p[si],s_YH2[si],s_YO2[si],s_YH2O[si]); 
  } 
 }
 else
 {  
  // boundary
  if(threadIdx.x < RAD)
  { 
   si = s_id - (2*threadIdx.x + 1); 
   di = id; 
           s_rho[si] = d_fl->getrho(di);
	   s_u[si] = -d_fl->getu(di); // minus sign for free slip wall
	   s_v[si] = d_fl->getv(di);
	   s_p[si] = d_fl->getp(di);
	   s_e[si] = d_fl->gete(di);
	   s_YH2[si] = d_fl->getYH2(di); 
	   s_YO2[si] = d_fl->getYO2(di); 
	   s_YH2O[si] = d_fl->getYH2O(di); 
	  
	   // for consistency 
	   s_e[si] = compute_e(s_rho[si],s_p[si],s_YH2[si],s_YO2[si],s_YH2O[si]);
  } 
 } 

 
 //__syncthreads(); 

//--------------------------------

 // right
 if(blockIdx.x < gridDim.x-1)
 {
  // interior
  if(threadIdx.x > blockDim.x-1-RAD)
  { 
   si = s_id + RAD;
   di = id + RAD; 
           s_rho[si] = d_fl->getrho(di);
	   s_u[si] = d_fl->getu(di);
	   s_v[si] = d_fl->getv(di);
	   s_p[si] = d_fl->getp(di);
	   s_e[si] = d_fl->gete(di);
	   s_YH2[si] = d_fl->getYH2(di); 
	   s_YO2[si] = d_fl->getYO2(di); 
	   s_YH2O[si] = d_fl->getYH2O(di); 
	  
	   // for consistency 
	   s_e[si] = compute_e(s_rho[si],s_p[si],s_YH2[si],s_YO2[si],s_YH2O[si]);
  } 
 }
 else
 {
  // boundary
  if(threadIdx.x > blockDim.x-1-RAD)
  { 
   si = s_id + RAD;
   di = id + blockDim.x - RAD - threadIdx.x + 1;
           s_rho[si] = d_fl->getrho(di);
	   s_u[si] = d_fl->getu(di);
	   s_v[si] = d_fl->getv(di);
	   s_p[si] = d_fl->getp(di);
	   s_e[si] = d_fl->gete(di);
	   s_YH2[si] = d_fl->getYH2(di); 
	   s_YO2[si] = d_fl->getYO2(di); 
	   s_YH2O[si] = d_fl->getYH2O(di); 
	  
	   // for consistency 
	   s_e[si] = compute_e(s_rho[si],s_p[si],s_YH2[si],s_YO2[si],s_YH2O[si]);
  }
 }  
  

 //__syncthreads(); 
  
 
//--------------------------------


 // top
 if(blockIdx.y > 0)
 {
  // interior
  if(threadIdx.y < RAD)
  { 
   si = s_id - RAD*s_x;
   di = id - RAD*blockDim.x*gridDim.x; 
           s_rho[si] = d_fl->getrho(di);
	   s_u[si] = d_fl->getu(di);
	   s_v[si] = d_fl->getv(di);
	   s_p[si] = d_fl->getp(di);
	   s_e[si] = d_fl->gete(di);
	   s_YH2[si] = d_fl->getYH2(di); 
	   s_YO2[si] = d_fl->getYO2(di); 
	   s_YH2O[si] = d_fl->getYH2O(di); 
	  
	   // for consistency 
	   s_e[si] = compute_e(s_rho[si],s_p[si],s_YH2[si],s_YO2[si],s_YH2O[si]);
  } 
 }
 else
 {  
  // boundary
  if(threadIdx.y < RAD)
  { 
   si = s_id - (2*threadIdx.y + 1)*s_x; 
   di = id; 
           s_rho[si] = d_fl->getrho(di);
	   s_u[si] = d_fl->getu(di);
	   s_v[si] = -d_fl->getv(di); // minus sign for free-slip wall
	   s_p[si] = d_fl->getp(di);
	   s_e[si] = d_fl->gete(di);
	   s_YH2[si] = d_fl->getYH2(di); 
	   s_YO2[si] = d_fl->getYO2(di); 
	   s_YH2O[si] = d_fl->getYH2O(di); 
	  
	   // for consistency 
	   s_e[si] = compute_e(s_rho[si],s_p[si],s_YH2[si],s_YO2[si],s_YH2O[si]);
  } 
 } 


 //__syncthreads(); 


//--------------------------------
 
 // bottom
 if(blockIdx.y < gridDim.y-1)
 {
  // interior
  if(threadIdx.y > blockDim.y-1-RAD)
  { 
   si = s_id + RAD*s_x;
   di = id + RAD*blockDim.x*gridDim.x; 
           s_rho[si] = d_fl->getrho(di);
	   s_u[si] = d_fl->getu(di);
	   s_v[si] = d_fl->getv(di);
	   s_p[si] = d_fl->getp(di);
	   s_e[si] = d_fl->gete(di);
	   s_YH2[si] = d_fl->getYH2(di); 
	   s_YO2[si] = d_fl->getYO2(di); 
	   s_YH2O[si] = d_fl->getYH2O(di); 
	  
	   // for consistency 
	   s_e[si] = compute_e(s_rho[si],s_p[si],s_YH2[si],s_YO2[si],s_YH2O[si]);
  } 
 }
 else
 {  
  // boundary
  if(threadIdx.y > blockDim.y-1-RAD)
  { 
   si = s_id + (2*(blockDim.y-1-threadIdx.y) + 1)*s_x; 
   di = id; 
           s_rho[si] = d_fl->getrho(di);
	   s_u[si] = d_fl->getu(di);
	   s_v[si] = -d_fl->getv(di); // minus sign for free-slip wall
	   s_p[si] = d_fl->getp(di);
	   s_e[si] = d_fl->gete(di);
	   s_YH2[si] = d_fl->getYH2(di); 
	   s_YO2[si] = d_fl->getYO2(di); 
	   s_YH2O[si] = d_fl->getYH2O(di); 
	  
	   // for consistency 
	   s_e[si] = compute_e(s_rho[si],s_p[si],s_YH2[si],s_YO2[si],s_YH2O[si]);
  } 
 } 
 
//--------------------------------

 //__syncthreads();


//--------------------------------
// compute fluxes
//--------------------------------  
// Rusanov Riemann solver

//--------------------------------
// x-flux

  // wave speed
  wm1 = abs(s_u[s_id-1]) + compute_sound(s_rho[s_id-1],s_p[s_id-1],s_YH2[s_id-1],s_YO2[s_id-1],s_YH2O[s_id-1]);
  w = abs(s_u[s_id]) + compute_sound(s_rho[s_id],s_p[s_id],s_YH2[s_id],s_YO2[s_id],s_YH2O[s_id]);
  wp1 = abs(s_u[s_id+1]) + compute_sound(s_rho[s_id+1],s_p[s_id+1],s_YH2[s_id+1],s_YO2[s_id+1],s_YH2O[s_id+1]);
  
  wavel = (wm1>w) ? wm1 : w;
  waver = (wp1>w) ? wp1 : w;

  // continuity
  consm1 = s_rho[s_id-1];
  cons = s_rho[s_id];
  consp1 = s_rho[s_id+1];

  fm1 = s_rho[s_id-1]*s_u[s_id-1];
  f = s_rho[s_id]*s_u[s_id];
  fp1 = s_rho[s_id+1]*s_u[s_id+1];

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdx[0] = (fluxr-fluxl)/dx;

  
  // x-momentum
  consm1 = s_rho[s_id-1]*s_u[s_id-1];
  cons = s_rho[s_id]*s_u[s_id];
  consp1 = s_rho[s_id+1]*s_u[s_id+1];

  fm1 = s_rho[s_id-1]*s_u[s_id-1]*s_u[s_id-1] + s_p[s_id-1];
  f = s_rho[s_id]*s_u[s_id]*s_u[s_id] + s_p[s_id];
  fp1 = s_rho[s_id+1]*s_u[s_id+1]*s_u[s_id+1] + s_p[s_id+1];

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdx[1] = (fluxr-fluxl)/dx;

  
  // y-momentum
  consm1 = s_rho[s_id-1]*s_v[s_id-1];
  cons = s_rho[s_id]*s_v[s_id];
  consp1 = s_rho[s_id+1]*s_v[s_id+1];

  fm1 = s_rho[s_id-1]*s_u[s_id-1]*s_v[s_id-1];
  f = s_rho[s_id]*s_u[s_id]*s_v[s_id];
  fp1 = s_rho[s_id+1]*s_u[s_id+1]*s_v[s_id+1];

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdx[2] = (fluxr-fluxl)/dx;
 
 
  // energy
  ke = 0.5*(s_u[s_id-1]*s_u[s_id-1] + s_v[s_id-1]*s_v[s_id-1]);
  consm1 = s_rho[s_id-1]*(s_e[s_id-1] + ke);
  fm1 = s_rho[s_id-1]*s_u[s_id-1]*(s_e[s_id-1] + ke) + s_p[s_id-1]*s_u[s_id-1];

  ke = 0.5*(s_u[s_id]*s_u[s_id] + s_v[s_id]*s_v[s_id]);
  cons = s_rho[s_id]*(s_e[s_id] + ke);
  f = s_rho[s_id]*s_u[s_id]*(s_e[s_id] + ke) + s_p[s_id]*s_u[s_id];

  ke = 0.5*(s_u[s_id+1]*s_u[s_id+1] + s_v[s_id+1]*s_v[s_id+1]);
  consp1 = s_rho[s_id+1]*(s_e[s_id+1] + ke);
  fp1 = s_rho[s_id+1]*s_u[s_id+1]*(s_e[s_id+1] + ke) + s_p[s_id+1]*s_u[s_id+1];  

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdx[3] = (fluxr-fluxl)/dx;


  // H2
  consm1 = s_rho[s_id-1]*s_YH2[s_id-1];
  cons = s_rho[s_id]*s_YH2[s_id];
  consp1 = s_rho[s_id+1]*s_YH2[s_id+1];

  fm1 = s_rho[s_id-1]*s_u[s_id-1]*s_YH2[s_id-1];
  f = s_rho[s_id]*s_u[s_id]*s_YH2[s_id];
  fp1 = s_rho[s_id+1]*s_u[s_id+1]*s_YH2[s_id+1];

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdx[4] = (fluxr-fluxl)/dx;


  // O2
  consm1 = s_rho[s_id-1]*s_YO2[s_id-1];
  cons = s_rho[s_id]*s_YO2[s_id];
  consp1 = s_rho[s_id+1]*s_YO2[s_id+1];

  fm1 = s_rho[s_id-1]*s_u[s_id-1]*s_YO2[s_id-1];
  f = s_rho[s_id]*s_u[s_id]*s_YO2[s_id];
  fp1 = s_rho[s_id+1]*s_u[s_id+1]*s_YO2[s_id+1];

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdx[5] = (fluxr-fluxl)/dx; 


  // H2O
  consm1 = s_rho[s_id-1]*s_YH2O[s_id-1];
  cons = s_rho[s_id]*s_YH2O[s_id];
  consp1 = s_rho[s_id+1]*s_YH2O[s_id+1];

  fm1 = s_rho[s_id-1]*s_u[s_id-1]*s_YH2O[s_id-1];
  f = s_rho[s_id]*s_u[s_id]*s_YH2O[s_id];
  fp1 = s_rho[s_id+1]*s_u[s_id+1]*s_YH2O[s_id+1];

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdx[6] = (fluxr-fluxl)/dx; 


//--------------------------------
// y-flux

  // wave speed
  wm1 = abs(s_v[s_id+s_x]) + compute_sound(s_rho[s_id+s_x],s_p[s_id+s_x],s_YH2[s_id+s_x],s_YO2[s_id+s_x],s_YH2O[s_id+s_x]);
  w = abs(s_v[s_id]) + compute_sound(s_rho[s_id],s_p[s_id],s_YH2[s_id],s_YO2[s_id],s_YH2O[s_id]);
  wp1 = abs(s_v[s_id-s_x]) + compute_sound(s_rho[s_id-s_x],s_p[s_id-s_x],s_YH2[s_id-s_x],s_YO2[s_id-s_x],s_YH2O[s_id-s_x]);
  
  wavel = (wm1>w) ? wm1 : w;
  waver = (wp1>w) ? wp1 : w;

  // continuity
  consm1 = s_rho[s_id+s_x];
  cons = s_rho[s_id];
  consp1 = s_rho[s_id-s_x];

  fm1 = s_rho[s_id+s_x]*s_v[s_id+s_x];
  f = s_rho[s_id]*s_v[s_id];
  fp1 = s_rho[s_id-s_x]*s_v[s_id-s_x];

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdy[0] = (fluxr-fluxl)/dy;

  
  // x-momentum
  consm1 = s_rho[s_id+s_x]*s_u[s_id+s_x];
  cons = s_rho[s_id]*s_u[s_id];
  consp1 = s_rho[s_id-s_x]*s_u[s_id-s_x];

  fm1 = s_rho[s_id+s_x]*s_u[s_id+s_x]*s_v[s_id+s_x];
  f = s_rho[s_id]*s_u[s_id]*s_v[s_id];
  fp1 = s_rho[s_id-s_x]*s_u[s_id-s_x]*s_v[s_id-s_x];

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdy[1] = (fluxr-fluxl)/dy;

  
  // y-momentum
  consm1 = s_rho[s_id+s_x]*s_v[s_id+s_x];
  cons = s_rho[s_id]*s_v[s_id];
  consp1 = s_rho[s_id-s_x]*s_v[s_id-s_x];

  fm1 = s_rho[s_id+s_x]*s_v[s_id+s_x]*s_v[s_id+s_x] + s_p[s_id+s_x];
  f = s_rho[s_id]*s_v[s_id]*s_v[s_id] + s_p[s_id];
  fp1 = s_rho[s_id-s_x]*s_v[s_id-s_x]*s_v[s_id-s_x] + s_p[s_id-s_x];

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdy[2] = (fluxr-fluxl)/dy;
 
 
  // energy
  ke = 0.5*(s_u[s_id+s_x]*s_u[s_id+s_x] + s_v[s_id+s_x]*s_v[s_id+s_x]);
  consm1 = s_rho[s_id+s_x]*(s_e[s_id+s_x] + ke);
  fm1 = s_rho[s_id+s_x]*s_v[s_id+s_x]*(s_e[s_id+s_x] + ke) + s_p[s_id+s_x]*s_v[s_id+s_x];

  ke = 0.5*(s_u[s_id]*s_u[s_id] + s_v[s_id]*s_v[s_id]);
  cons = s_rho[s_id]*(s_e[s_id] + ke);
  f = s_rho[s_id]*s_v[s_id]*(s_e[s_id] + ke) + s_p[s_id]*s_v[s_id];

  ke = 0.5*(s_u[s_id-s_x]*s_u[s_id-s_x] + s_v[s_id-s_x]*s_v[s_id-s_x]);
  consp1 = s_rho[s_id-s_x]*(s_e[s_id-s_x] + ke);
  fp1 = s_rho[s_id-s_x]*s_v[s_id-s_x]*(s_e[s_id-s_x] + ke) + s_p[s_id-s_x]*s_v[s_id-s_x];  

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdy[3] = (fluxr-fluxl)/dy;


  // H2
  consm1 = s_rho[s_id+s_x]*s_YH2[s_id+s_x];
  cons = s_rho[s_id]*s_YH2[s_id];
  consp1 = s_rho[s_id-s_x]*s_YH2[s_id-s_x];

  fm1 = s_rho[s_id+s_x]*s_v[s_id+s_x]*s_YH2[s_id+s_x];
  f = s_rho[s_id]*s_v[s_id]*s_YH2[s_id];
  fp1 = s_rho[s_id-s_x]*s_v[s_id-s_x]*s_YH2[s_id-s_x];

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdy[4] = (fluxr-fluxl)/dy;


  // O2
  consm1 = s_rho[s_id+s_x]*s_YO2[s_id+s_x];
  cons = s_rho[s_id]*s_YO2[s_id];
  consp1 = s_rho[s_id-s_x]*s_YO2[s_id-s_x];

  fm1 = s_rho[s_id+s_x]*s_v[s_id+s_x]*s_YO2[s_id+s_x];
  f = s_rho[s_id]*s_v[s_id]*s_YO2[s_id];
  fp1 = s_rho[s_id-s_x]*s_v[s_id-s_x]*s_YO2[s_id-s_x];

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdy[5] = (fluxr-fluxl)/dy; 


  // H2O
  consm1 = s_rho[s_id+s_x]*s_YH2O[s_id+s_x];
  cons = s_rho[s_id]*s_YH2O[s_id];
  consp1 = s_rho[s_id-s_x]*s_YH2O[s_id-s_x];

  fm1 = s_rho[s_id+s_x]*s_v[s_id+s_x]*s_YH2O[s_id+s_x];
  f = s_rho[s_id]*s_v[s_id]*s_YH2O[s_id];
  fp1 = s_rho[s_id-s_x]*s_v[s_id-s_x]*s_YH2O[s_id-s_x];

  fluxr = 0.5*(f+fp1) - 0.5*waver*(consp1-cons);
  fluxl = 0.5*(fm1+f) - 0.5*wavel*(cons-consm1);
  dfdy[6] = (fluxr-fluxl)/dy; 

  
 
//--------------------------------
// update
  
  for (l=0; l<neqns; l++)
  {
   dQ[l] = -dt*(dfdx[l] + dfdy[l]); 
  }

//--------------------------------
// combustion
// H2 + 1/2 O2 -> H2O

  if(combustion == true)
  {
  
   // conc in mol/cc
   concH2 = d_fl->getrho(id)*d_fl->getYH2(id)/1000.0/MWH2;
   concO2 = d_fl->getrho(id)*d_fl->getYO2(id)/1000.0/MWO2;
   T = compute_T(d_fl->getrho(id),d_fl->getp(id),d_fl->getYH2(id),d_fl->getYO2(id),d_fl->getYH2O(id));
   
   // ignition temperature 
   if(T >= 500.0)
   {
    k_react = Arr*exp(-Eact/T); 
    RR = k_react*concH2*sqrt(concO2); // mol/cc/sec
   }
   else
   {
    RR = 0.0;
   }  

 
   source[0] = -RR*MWH2*1000.0; // H2
   source[1] = -0.5*RR*MWO2*1000.0; // O2
   source[2] = RR*MWH2O*1000.0; // H2O
   source[3] = RR*Qheat*(1.0e6); // J/cc/sec -> J/m3/sec  // energy eqn

   dQ[3] += source[3]*dt; // energy eqn
   dQ[4] += source[0]*dt;  // H2
   dQ[5] += source[1]*dt; // O2
   dQ[6] += source[2]*dt; // H2O

  }
//------------------------------------------------ 


  new_q[0] = d_Q->getq1(id) + dQ[0];
  new_q[1] = d_Q->getq2(id) + dQ[1];
  new_q[2] = d_Q->getq3(id) + dQ[2];
  new_q[3] = d_Q->getq4(id) + dQ[3];
  new_q[4] = d_Q->getq5(id) + dQ[4];
  new_q[5] = d_Q->getq6(id) + dQ[5];
  new_q[6] = d_Q->getq7(id) + dQ[6];


  // mass fractions
  new_yh2 = new_q[4]/new_q[0];
  new_yo2 = new_q[5]/new_q[0];
  new_yh2o = new_q[6]/new_q[0];


  // make sure mass fractions >= 0
  new_yh2 = (new_yh2 > 0.0)? new_yh2 : 0.0;
  new_yo2 = (new_yo2 > 0.0)? new_yo2 : 0.0;
  new_yh2o = (new_yh2o > 0.0)? new_yh2o : 0.0;
   
  // make sure mass fractions <= 1
  new_yh2 = (new_yh2 < 1.0)? new_yh2 : 1.0;
  new_yo2 = (new_yo2 < 1.0)? new_yo2 : 1.0;
  new_yh2o = (new_yh2o < 1.0)? new_yh2o : 1.0;  


  sumYk = new_yh2 + new_yo2 + new_yh2o;
  new_yh2 = new_yh2/sumYk;
  new_yo2 = new_yo2/sumYk;
  new_yh2o = new_yh2o/sumYk;

  new_q[4] = new_q[0]*new_yh2;
  new_q[5] = new_q[0]*new_yo2;
  new_q[6] = new_q[0]*new_yh2o;

//--------------------------------

  d_Q->editQ(id,new_q[0],new_q[1],new_q[2],new_q[3],new_q[4],new_q[5],new_q[6]);


  // cons2prim 
  
  new_rho = d_Q->getq1(id);
  new_u = d_Q->getq2(id)/new_rho;
  new_v = d_Q->getq3(id)/new_rho;

  ke = 0.5*(new_u*new_u + new_v*new_v);
  new_e = d_Q->getq4(id)/new_rho - ke;
  
  new_yh2 = d_Q->getq5(id)/new_rho;
  new_yo2 = d_Q->getq6(id)/new_rho;
  new_yh2o = d_Q->getq7(id)/new_rho;

  new_p = compute_p(new_rho,new_e,new_yh2,new_yo2,new_yh2o);

//-------------------------------- 

  d_fl->setfluid(id,new_rho,new_u,new_v,new_p,new_yh2,new_yo2,new_yh2o);


  __syncthreads();


}


//-----------------------------------------------------------
  

// function kadvance
void kadvance(fluid *fl, conservative *Q)
{
 fluid *d_fl = new fluid;
 conservative *d_Q = new conservative;

 cudaMalloc(&d_fl,sizeof(fluid));
 cudaMalloc(&d_Q,sizeof(conservative));
 cudaMemcpy(d_fl,fl,sizeof(fluid),cudaMemcpyHostToDevice);
 cudaMemcpy(d_Q,Q,sizeof(conservative),cudaMemcpyHostToDevice);

 const dim3 blockSize(TPB, TPB);
 const dim3 gridSize = dim3((Nx + TPB - 1)/TPB, (Ny + TPB - 1)/TPB);
 
 for (int n = 0; n<nmax; n++)
 {  
 advanceKernel<<<gridSize, blockSize>>>(d_fl,d_Q);
 }

 cudaMemcpy(Q,d_Q,sizeof(conservative),cudaMemcpyDeviceToHost);
 cudaMemcpy(fl,d_fl,sizeof(fluid),cudaMemcpyDeviceToHost); 

 cudaFree(d_fl);
 cudaFree(d_Q); 
}

//-----------------------------------------------------------
