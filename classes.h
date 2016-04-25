#ifndef CLASSES_H
#define CLASSES_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include <iostream>
#include <math.h>
#include "consts.h"

class fluid
{

 private:
  double rho[Nx*Ny];
  double u[Nx*Ny];
  double v[Nx*Ny];
  double e[Nx*Ny];
  double p[Nx*Ny];
  double c[Nx*Ny];
  double YH2[Nx*Ny];
  double YO2[Nx*Ny];
  double YH2O[Nx*Ny];
 public:

  // constructor
  CUDA_CALLABLE_MEMBER fluid()
  {
   init();
  } 

  // destructor
  CUDA_CALLABLE_MEMBER ~fluid()
  {
  }  
 
  // function init
  CUDA_CALLABLE_MEMBER void init()
  {
   int i, j, k;
   k = -1;
   for (j=0; j<Ny; j++)
   {
    for (i=0; i<Nx; i++)
    {
     k++;  
     rho[k] = rhoref;      
     p[k] = pref;
     u[k] = 0.0;
     v[k] = 0.0;
     
     YH2[k] = yh2ref;
     YO2[k] = yo2ref; 
     YH2O[k] = yh2oref; 

     e[k] = compute_e(rho[k],p[k],YH2[k],YO2[k],YH2O[k]);

     c[k] = sound(rho[k],p[k],YH2[k],YO2[k],YH2O[k]);   
    }
   }
  }

  // copy constructor
  CUDA_CALLABLE_MEMBER fluid(fluid &f)
  {
   int i, j, k;
   k = -1;
   for (j=0; j<Ny; j++)
   {
    for (i=0; i<Nx; i++)
    {
     k++;
     this->rho[k] = f.rho[k];
     this->u[k] = f.u[k];
     this->v[k] = f.v[k];
     this->p[k] = f.p[k];
     this->e[k] = f.e[k];
     this->c[k] = f.c[k];
     this->YH2[k] = f.YH2[k];
     this->YO2[k] = f.YO2[k];
     this->YH2O[k] = f.YH2O[k];
    }
   }
  } 

  // function getrho
  CUDA_CALLABLE_MEMBER double getrho(int k)
  {
   return rho[k];
  }

  // function getu
  CUDA_CALLABLE_MEMBER double getu(int k)
  {
   return u[k];
  }

  // function getv
  CUDA_CALLABLE_MEMBER double getv(int k)
  {
   return v[k];
  }

  // function gete
  CUDA_CALLABLE_MEMBER double gete(int k)
  {
   return e[k];
  }

  // function getp
  CUDA_CALLABLE_MEMBER double getp(int k)
  {
   return p[k];
  }

  // function getYH2
  CUDA_CALLABLE_MEMBER double getYH2(int k)
  {
   return YH2[k];
  }
  
  // function getYO2
  CUDA_CALLABLE_MEMBER double getYO2(int k)
  {
   return YO2[k];
  }

  // function getYH2O
  CUDA_CALLABLE_MEMBER double getYH2O(int k)
  {
   return YH2O[k];
  }

  // function sound
  CUDA_CALLABLE_MEMBER double sound(double rho1, double p1, double yh2, double yo2, double yh2o)
  {
   double cpmix, cvmix, gammix;
   cpmix = yh2*CpH2 + yo2*CpO2 + yh2o*CpH2O;
   cvmix = yh2*(CpH2-Runiv) + yo2*(CpO2-Runiv) + yh2o*(CpH2O-Runiv);
   gammix = cpmix/cvmix;
   return sqrt(gammix*p1/rho1);
  }

  // function compute_e
  CUDA_CALLABLE_MEMBER double compute_e(double rho1, double p1, double yh2, double yo2, double yh2o)
  {
   double cvmix, mw, T;
   cvmix = yh2*(CpH2-Runiv) + yo2*(CpO2-Runiv) + yh2o*(CpH2O-Runiv);
   mw = 1.0/(yh2/MWH2 + yo2/MWO2 + yh2o/MWH2O); 
   T = p1/rho1/(1000.0*Runiv/mw);
   return (1000.0*cvmix/mw*T);
  }

  
  

  // function setfluid
  CUDA_CALLABLE_MEMBER void setfluid(int k, double rho1, double u1, double v1, double p1, double yh2, double yo2, double yh2o)
  {
   this->rho[k] = rho1;
   this->u[k] = u1;
   this->v[k] = v1;
   this->p[k] = p1;
   this->YH2[k] = yh2;
   this->YO2[k] = yo2;
   this->YH2O[k] = yh2o;
   this->e[k] = compute_e(rho1,p1,yh2,yo2,yh2o);
   this->c[k] = sound(rho1,p1,yh2,yo2,yh2o);
  }

};



class conservative
{

 private:
  double Q1[Nx*Ny];
  double Q2[Nx*Ny];
  double Q3[Nx*Ny];
  double Q4[Nx*Ny];
  double Q5[Nx*Ny];
  double Q6[Nx*Ny];
  double Q7[Nx*Ny]; 
 public:

  // constructor
  CUDA_CALLABLE_MEMBER conservative()
  {
   conservativeInit();
  } 

  // destructor
  CUDA_CALLABLE_MEMBER ~conservative()
  {
  }  
 

  // function conservativeInit
  CUDA_CALLABLE_MEMBER void conservativeInit()
  {
   int i, j, k;
   double eref = compute_e(rhoref,pref,yh2ref,yo2ref,yh2oref);
   k = -1;
   for (j=0; j<Ny; j++)
   {
    for (i=0; i<Nx; i++)
    {
     k++;  
     Q1[k] = rhoref;      
     Q2[k] = 0.0;
     Q3[k] = 0.0;
     Q4[k] = rhoref*eref;
     Q5[k] = rhoref*yh2ref;
     Q6[k] = rhoref*yo2ref;
     Q7[k] = rhoref*yh2oref;  
    }
   }
  }

  // function compute_e
  CUDA_CALLABLE_MEMBER double compute_e(double rho1, double p1, double yh2, double yo2, double yh2o)
  {
   double cvmix, mw, T;
   cvmix = yh2*(CpH2-Runiv) + yo2*(CpO2-Runiv) + yh2o*(CpH2O-Runiv);
   mw = 1.0/(yh2/MWH2 + yo2/MWO2 + yh2o/MWH2O);
   T = p1/rho1/(1000.0*Runiv/mw);
   return (1000.0*cvmix/mw*T);
  }


  // function edit 
  CUDA_CALLABLE_MEMBER void editQ(int k, double q1, double q2, double q3, double q4, double q5, double q6, double q7)
  {
   Q1[k] = q1;
   Q2[k] = q2;
   Q3[k] = q3;
   Q4[k] = q4;
   Q5[k] = q5;
   Q6[k] = q6;
   Q7[k] = q7;
  }

  CUDA_CALLABLE_MEMBER double getq1(int k)
  {
   return Q1[k];
  }

  CUDA_CALLABLE_MEMBER double getq2(int k)
  {
   return Q2[k];
  }

  CUDA_CALLABLE_MEMBER double getq3(int k)
  {
   return Q3[k];
  }

  CUDA_CALLABLE_MEMBER double getq4(int k)
  {
   return Q4[k];
  }

  CUDA_CALLABLE_MEMBER double getq5(int k)
  {
   return Q5[k];
  }

  CUDA_CALLABLE_MEMBER double getq6(int k)
  {
   return Q6[k];
  }

  CUDA_CALLABLE_MEMBER double getq7(int k)
  {
   return Q7[k];
  }



  // copy constructor
  CUDA_CALLABLE_MEMBER conservative(conservative &cons)
  {
   int i, j, k;
   k = -1;
   for (j=0; j<Ny; j++)
   {
    for (i=0; i<Nx; i++)
    {
     k++;
     this->Q1[k] = cons.Q1[k];
     this->Q2[k] = cons.Q2[k];
     this->Q3[k] = cons.Q3[k];
     this->Q4[k] = cons.Q4[k];
     this->Q5[k] = cons.Q5[k];
     this->Q6[k] = cons.Q6[k];
     this->Q7[k] = cons.Q7[k];
    }
   }
  } 

};


#endif


