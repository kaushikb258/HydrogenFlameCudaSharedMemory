#ifndef FUNCS_H
#define FUNCS_H

void prim2cons(int, fluid*, conservative*); 
void advance(fluid*, conservative*);
void out_vtk(fluid*);

#endif
