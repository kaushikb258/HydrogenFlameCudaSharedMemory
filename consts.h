#ifndef CONSTS_H
#define CONSTS_H

// reference quantities

const double rhoref = 0.48;
const double pref = 1.0e5;
const double yh2ref = 2.0/18.0;
const double yo2ref = 16.0/18.0;
const double yh2oref = 0.0;

// grid

const int Nx = 1536;
const int Ny = 192;
const int nmax = 14000; // # time steps
const double dx = 1.0e-4;
const double dy = dx;
const double dt = 2.0e-9;
const int neqns = 7;

// hot source

const double Tsource = 1200.0;


// thermodynamics

const double MWH2 = 2.0;
const double MWO2 = 32.0;
const double MWH2O = 18.0;

const double CpH2 = 28.82;
const double CpO2 = 29.35;
const double CpH2O = 35.22;

const double Runiv = 8.314;


// combustion

const double Eact = 4500.0;
const double Arr = 2.5e12;
const double Qheat = Runiv*9800.0;
const bool combustion = true;

#endif
