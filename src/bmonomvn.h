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
 * MultiVariate Normal cumulative sum collector for 
 * collecting the posterior mean and variance of
 * the mean vector and covariance matrix 
 */

typedef struct MVNsum
{
  unsigned int m;           /* dimension of these matrices and vectors */
  unsigned int T;           /* number of accumulated samples in the sum */
  double *mu;               /* cumulative mean vector */
  double **S;               /* cumulative covariance matrix */
} MVNsum;


typedef struct QPsamp
{
  unsigned int m;           /* number of weights/rows in Amat/cols in W */
  unsigned int T;           /* total number of samples/rows in W */
  double *S_copy;           /* copy of S for qpgen2 */
  double *dvec;             /* vector of length m, either mu-samples or zeros */
  double *dvec_copy;        /* copy of dvec for qpgen2 */
  bool dmu;                 /* indicates if d should be replaced with mu-samps */
  double *Amat;             /* matrix of linear constraints t(A) %*% b >= b0 */
  double *Amat_copy;        /* copy of Amat for qpgen2 and mu_constr */
  double *b0;               /* vector of linear constraints t(A) %*% b >= b0 */
  double *b0_copy;          /* copy of b0 for qpgen2 mu_constr */
  int *mu_c;                /* which rows of Amat should mu be copied into */
  unsigned int mu_c_len;    /* length(mu_constr) */
  unsigned int q;           /* number of columns in Amat */
  int* iact;                /* workspace of length q */
  unsigned int meq;         /* first meq constraints are treated as equality */
  double *work;             /* workspace of length 2*n + r*(r+5)/2 + 2*q + 1 */
  double **W;               /* T-x-m samples of weights (b as above) */
} QPsamp;


/*
 * CLASS for the implementation of the monomvn
 * algorithm with Bayesian regressions, e.g., the
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

  /* large normed design matrix used in regressions */
  double *Xnorm;            /* normalization constants for the cols of X */
  double *Xmean;            /* mean of Xorig for centering */
  double **X;               /* the (normd and centered) design matrix */

  /* model parameters */
  double *mu;                /* estimated mean vector (in round t) */
  double **S;                /* estimated covariance matrix (in round t) */

  /* for the Bayesian regressions */
  Blasso **blasso;           /* pointer to M Bayesian lasso regressions,
                                one for each column: n=n[i], m=i */

  /* printing */
  unsigned int verb;         /* verbosity argument */

  /* Blasso regression model parameters, used for all i */
  int m;                     /* number of non-zero components of beta */
  double mu_s;               /* intercept term in the regression */
  double lambda2;            /* lasso penalty parameter */
  double s2;                 /* regression error variance */
  double *beta;              /* regression coefficients */
  double *tau2i;             /* latent vector of (inverse-) diagonal 
                                component of beta prior */

  /* posterior probability */
  double lpost_bl;           /* regression (each i) log posterior probability */
  double lpost_map;          /* best total log posterior probability */
  int which_map;             /* the time index giving the MAP sample */
  
  /* utility vectors for addy, used for all i */
  double *s21;               /* utility vector for calcing S from beta */
  double *y;                 /* for (temporarily) storing columns of y */
  
  /* for printing traces */
  FILE *trace_mu;            /* traces of the mean */
  FILE *trace_S;             /* traces of the covariance */
  FILE **trace_lasso;        /* traces of the individual regressions */

  /* for collecting means and variances -- allocated externally */
  MVNsum *mom1;              /* retains the sum of mu and S samples */
  double *lambda2_sum;       /* retains the sum of lambda2 samples */
  double *m_sum;             /* retains the sum of m samples */
  MVNsum *mom2;              /* retans the sum of mu^2 and S^2 samples */
  MVNsum *map;               /* Maximum a' Posteriori mu and S */

  /* for collecting samples of w from via Quadratic Programming */
  QPsamp *qps;               /* as above */

 protected:

  double Draw(const unsigned int thin, const bool economize, const bool burnin);

 public:

  /* constructors and destructors */
  Bmonomvn(const unsigned int M, const unsigned int N, double **Y, int *n,
	   const double p, const unsigned int verb, const bool trace);
  ~Bmonomvn();

  /* Initialization */
  void InitBlassos(const unsigned int method, const unsigned int RJm, 
		   const bool capm, double *mu_start, double ** S_start, 
		   int *ncomp_start, double *lambda_start, const double mprior,
		   const double r, const double delta, const bool rao_s2, 
		   const bool economy, const bool trace);

  /* sampling from the posterior distribution */
  void Rounds(const unsigned int T, const unsigned int thin, 
	      const bool economy, const bool burnin);

  /* printing and tracing */
  double LpostMAP(int *which);
  void PrintRegressions(FILE *outfile);
  void InitTrace(unsigned int m);
  void PrintTrace(unsigned int m);
  void Methods(int *methods);
  void Thin(const unsigned int thin, int *thin_out);
  int Verb(void);

  /* setting pointers to allocated memory */
  void SetSums(MVNsum *mom1, MVNsum *mom2, double *lambda2_sum, double *m_sum, 
	       MVNsum *map);
  void SetQPsamp(QPsamp *qps);
};


/* functions used on MVNsum structures for caluclating 
   means and variances from Monte Carlo samples of mu and S */
MVNsum* new_MVNsum_R(const unsigned int m, double* mu, double* S);
void delete_MVNsum_R(MVNsum *mvnsum);
void MVN_add(MVNsum *mom1, double *mu, double **S, const unsigned int m);
void MVN_add2(MVNsum *mom2, double *mu, double **S, const unsigned int m);
void MVN_mean(MVNsum *mom1, const unsigned int T);
void MVN_var(MVNsum *mom2, MVNsum *mean, const unsigned int T);
void MVN_copy(MVNsum *map, double *mu, double **S, const unsigned int m);

/* functions used on QPsamp structures for Quadratic Programming */
QPsamp* new_QPsamp_R(const unsigned int m, const unsigned int T, 
		     double *dvec, const bool dmu, double *Amat, double *b0, 
		     const unsigned int q, const unsigned int meq, 
		     double *w);
void delete_QPsamp_R(QPsamp *qps);
void QPsolve(QPsamp *qps, const unsigned int t, const unsigned int m,
	     double *mu, double **S);

/* getting the regression coefficients from a mean vector and covariance matrix */
void get_regress(const unsigned int m, double *mu, double *s21, double **s11, 
		 const unsigned int ncomp, double *mu_out, double *beta_out, 
		 double *s2_out);

#endif
