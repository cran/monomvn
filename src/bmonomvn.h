/******************************************************************************** 
 *
 * Estimation for Multivariate Normal Data with Monotone Missingness
 * Copyright (C) 2007, University of Cambridge
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Questions? Contact Robert B. Gramacy (bobby@statslab.cam.ac.uk)
 *
 ********************************************************************************/


#ifndef __BMONOMVN_H__
#define __BMONOMVN_H__ 

#include <fstream>
#include "blasso.h"

/*
 * CLASS for the implementation of the monomvn
 * algorithm with bayesian regressions, e.g., the
 * Bayesian Lasso, together with function for interfacing
 * with R in plain C.
 */

class Bmonomvn
{
 private:

  /* inputs */
  unsigned int M;
  unsigned int N;
  int *n;
  double **Y;
  double p;

  /* model parameters */
  double *mu;
  double **S;

  /* for the Bayesian regressions */
  Blasso **blasso;

  /* printing */
  unsigned int verb;
  
  /* utility vectors for addy */
  double mu_s, lambda2, s2;
  double *beta;
  double *tau2i;
  double *s21;
  
  /* for printing traces */
  FILE *trace_mu;
  FILE *trace_S;
  FILE **trace_lasso;

 protected:

 public:

  /* means */
  double *mu_sum;
  double **S_sum;
  double *lambda2_sum;

  /* variances */
  double *mu2_sum;

  /* constructors and destructors */
  Bmonomvn(const unsigned int M, const unsigned int N, double **Y, int *n,
	   const double p, const double r, const double delta, 
	   const bool rao_S2, const unsigned int verb, const bool trace);
  ~Bmonomvn();

  /* sampling from the posterior distribution */
  void Draw(const unsigned int thin, const bool burnin);
  void Rounds(const unsigned int T, const unsigned int thin, 
	      const bool burnin);

  /* printing and tracing */
  void PrintRegressions(FILE *outfile);
  void InitTrace(unsigned int m);
  void PrintTrace(unsigned int m);
};


#endif
