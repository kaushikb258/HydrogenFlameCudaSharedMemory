#include <iostream>
#include <math.h>
#include "consts.h"
#include "classes.h"
#include "funcs.h"
using namespace std;

int main()
{
 fluid *fl = new fluid; 
 conservative *Q = new conservative;

 prim2cons(0,fl,Q);  

 advance(fl,Q); 
    
 out_vtk(fl); 

 delete fl;
 delete Q;
}
