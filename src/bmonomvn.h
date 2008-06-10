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
  unsigned int M;            /* ncol(Y) */
  unsigned int N;            /* nrow(Y) */
  int *n;                    /* number of non-NA in each col of Y */
  double **Y;                /* the data matrix */
  double p;                  /* the parsimony proportion */

  /* model parameters */
  double *mu;                /* estimated mean vector (in round t) */
  double **S;                /* estimated covariance matrix (in round t) */

  /* for the Bayesian regressions */
  Blasso **blasso;           /* pointer to M Bayesian lasso regressions,
                                one for each column: n=n[i], m=i */

  /* printing */
  unsigned int verb;         /* verbosity argument */
  
  /* utility vectors for addy, used for all i */
  int m;                     /* number of non-zero components of beta */
  double mu_s;               /* intercept term in the regression */
  double lambda2;            /* lasso penalty parameter */
  double s2;                 /* regression error variance */
  double lpost;              /* regression log posterior probability */
  double *beta;              /* regression coefficients */
  double *tau2i;             /* latent vector of (inverse-) diagonal 
                                component of beta prior */
  double *s21;               /* utility vector for calcing S from beta */
  
  /* for printing traces */
  FILE *trace_mu;            /* traces of the mean */
  FILE *trace_S;             /* traces of the covariance */
  FILE **trace_lasso;        /* traces of the individual regressions */

 protected:

 public:

  /* means */
  double *mu_sum;            /* retains the sum of mu samples  */
  double **S_sum;            /* retains the sum of S samples */
  double *lambda2_sum;       /* retains the sum of lambda2 samples */
  double *m_sum;             /* retains the sum of m samples */

  /* variances */
  double *mu2_sum;           /* retains the sum of mu^2 samples */

  /* constructors and destructors */
  Bmonomvn(const unsigned int M, const unsigned int N, double **Y, int *n,
	   const double p, const unsigned int RJ, const bool capm, 
	   const double r, const double delta, const bool rao_S2, 
	   const unsigned int verb, const bool trace);
  ~Bmonomvn();

  /* sampling from the posterior distribution */
  void Draw(const unsigned int thin, const bool burnin);
  void Rounds(const unsigned int T, const unsigned int thin, 
	      const bool burnin);

  /* printing and tracing */
  void PrintRegressions(FILE *outfile);
  void InitTrace(unsigned int m);
  void PrintTrace(unsigned int m);
  void Methods(int *methods);
};


#endif
