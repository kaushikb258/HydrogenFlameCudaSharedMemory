#include <iostream>
#include <math.h>
#include <string>
#include <sstream>
#include <fstream>
#include "consts.h"
#include "classes.h"
#include "funcs.h"
#include "kernel.h"
using namespace std;


double compute__e(double rho1, double p1, double yh2, double yo2, double yh2o)
{
   double cvmix, mw, T;
   cvmix = yh2*(CpH2-Runiv) + yo2*(CpO2-Runiv) + yh2o*(CpH2O-Runiv);
   mw = 1.0/(yh2/MWH2 + yo2/MWO2 + yh2o/MWH2O); 
   T = p1/rho1/(1000.0*Runiv/mw);
   return (1000.0*cvmix/mw*T);
}



void prim2cons(int init, fluid *fl, conservative *Q)
{

 int i, j, k;
 double rho1, u1, v1, e1, yh2, yo2, yh2o, ke, mw, p1;
 double q1, q2, q3, q4, q5, q6, q7; 
 double xx, x_c, y_c, r_c1, r_c2;

 const double xf = 20.0; 
 int abc;


 k = -1;

 for (j=0; j<Ny; j++)
 {
  for (i=0; i<Nx; i++)
  {
   k++;
   rho1 = fl->getrho(k);
   u1 = fl->getu(k);
   v1 = fl->getv(k);
   e1 = fl->gete(k);
   yh2 = fl->getYH2(k);
   yo2 = fl->getYO2(k);
   yh2o = fl->getYH2O(k); 

   if(init == 0)
   {
    xx = (double) (i);
    x_c = xx - xf ;
    y_c = (double) (j) - Ny/4;
    r_c1 = sqrt(x_c*x_c + y_c*y_c);
    y_c = (double) (j) - 3*Ny/4;
    r_c2 = sqrt(x_c*x_c + y_c*y_c);

    abc = 0;

    if(xx <= xf)
    { 
     abc = 1;
    } 
    if(r_c1 <= 5.0 || r_c2 <= 5.0)
    {   
     abc = 1;
    }   

    if(abc == 1)
    {
     p1 = 2.0*pref;
     mw = 1.0/(yh2ref/MWH2 + yo2ref/MWO2 + yh2oref/MWH2O);
     rho1 = p1/Tsource/(1000.0*Runiv/mw);
     e1 = compute__e(rho1,p1,yh2ref,yo2ref,yh2oref);
    }
   }


   q1 = rho1;
   q2 = rho1*u1;
   q3 = rho1*v1;
   ke = 0.5*(u1*u1 + v1*v1);
   q4 = rho1*(e1+ke);
   q5 = rho1*yh2;
   q6 = rho1*yo2;
   q7 = rho1*yh2o;

   Q->editQ(k,q1,q2,q3,q4,q5,q6,q7);
  }
 }

} 



void advance(fluid *fl, conservative *Q)
{
 kadvance(fl,Q);
}

void out_vtk(fluid *fl)
{
 string filename = "output_paraview.vtk";
 string title = "H2O2Flame"; 
 string cell_size_string, node_num_string;
 int node;
 int i, j, k, n;
 stringstream s_node_num, s_cells, s_imax, s_jmax, s_kmax;
 ofstream f;
 int Nz = 1;

 s_node_num << (Nx*Ny);
 s_cells << (Nx-1)*(Ny-1);
 s_imax << Nx;
 s_jmax << Ny;
 s_kmax << Nz;

 // initialize coordinates
 double coord[Nx*Ny*3];
 
 n = 0;
 for (j=0; j<Ny; j++)
 {
 for (i=0; i<Nx; i++)
 {
  coord[n] = ((double) (i) + 0.5)*dx; //x
  coord[n+1] = (double)(Ny)*dy - ((double) (j) + 0.5)*dy; //y
  coord[n+2] = 0.0; //z = 0 as 2D grid
  n += 3;
 }
 }
 

 f.open (filename.c_str(),ios_base::out);
 f<< "# vtk DataFile Version 2.0\n";
 f<< title<<"\n";
 f<< "ASCII\n";
 f<< "DATASET STRUCTURED_GRID\n";
 f<< "DIMENSIONS "<<"\t"<<s_imax.str()<<"\t\t"<<s_jmax.str()<<"\t\t"<<s_kmax.str()<<"\n";
 f<< "POINTS "<<"\t"<<s_node_num.str()<<"\t"<<"double\n";

 n = 0;
 for (j=0; j<Ny; j++)
 {
 for (i=0; i<Nx; i++)
 { 
  f<<coord[n]<<" "<<coord[n+1]<<" "<<coord[n+2]<<"\n";
  n += 3;
 }
 }

 f<< "CELL_DATA "<<"\t"<<s_cells.str()<<"\n";
 f<< "POINT_DATA "<<"\t"<<s_node_num.str()<<"\n";
 
 f<< "SCALARS rho double \n";
 f<< "LOOKUP_TABLE default \n";
 for (j=0; j<Ny; j++)
 {
 for (i=0; i<Nx; i++)
 {
  f<< fl->getrho(i+j*Nx)<<" ";
 } 
 }
 f<<"\n";

 f<< "SCALARS u double \n";
 f<< "LOOKUP_TABLE default \n";
 for (j=0; j<Ny; j++)
 {
 for (i=0; i<Nx; i++)
 {
  f<< fl->getu(i+j*Nx)<<" ";
 }
 }
 f<<"\n";
 
 f<< "SCALARS v double \n";
 f<< "LOOKUP_TABLE default \n";
 for (j=0; j<Ny; j++)
 {
 for (i=0; i<Nx; i++)
 {
  f<< fl->getv(i+j*Nx)<<" ";
 }
 }
 f<<"\n"; 

 f<< "SCALARS H2 double \n";
 f<< "LOOKUP_TABLE default \n";
 for (j=0; j<Ny; j++)
 {
 for (i=0; i<Nx; i++)
 {
  f<< fl->getYH2(i+j*Nx)<<" ";
 }
 }
 f<<"\n";

 f<< "SCALARS O2 double \n";
 f<< "LOOKUP_TABLE default \n";
 for (j=0; j<Ny; j++)
 {
 for (i=0; i<Nx; i++)
 {
  f<< fl->getYO2(i+j*Nx)<<" ";
 }
 }
 f<<"\n";

 f<< "SCALARS H2O double \n";
 f<< "LOOKUP_TABLE default \n";
 for (j=0; j<Ny; j++)
 {
 for (i=0; i<Nx; i++)
 {
  f<< fl->getYH2O(i+j*Nx)<<" ";
 }
 }
 f<<"\n";
 
 f<< "SCALARS e double \n";
 f<< "LOOKUP_TABLE default \n";
 for (j=0; j<Ny; j++)
 {
 for (i=0; i<Nx; i++)
 {
  f<< fl->gete(i+j*Nx)<<" ";
 }
 }
 f<<"\n"; 


 f.close();

} 
