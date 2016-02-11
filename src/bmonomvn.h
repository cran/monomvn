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
#include "ustructs.h"


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
  double **Y;                /* the data matrix */
  int *n;                    /* number of non-NA in each col of Y */
  Rmiss *R;                  /* the missingness pattern structure */
  // int *n2;                   /* number of R=2's in each col of R for DA */
  double p;                  /* the parsimony proportion */

  /* large normed design matrix used in regressions */
  double *Xnorm;            /* normalization constants for the cols of X */
  double *Xmean;            /* mean of Xorig for centering */

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
  double gam;                /* NG prior gamma parameter */
  double s2;                 /* regression error variance */
  double *beta;              /* regression coefficients */
  double *tau2i;             /* latent vector of (inverse-) diagonal 
                                component of beta prior */
  double *omega2;            /* diagonal of the covariance matrix D of Y~N(Xb,s2*D) */
  double theta;              /* Exp rate parameter to omegas for St errors */
  double nu;                 /* degrees of freedom parameter in Student-t model */

  bool onenu;                /* indicates if we should pool all the nu's from Blasso */
  double pi;                 /* prior parameter p in Bin(m|M,p) for model order m
                                or indicates Unif[0,...,Mmax] when p=0 */

  /* posterior probability & likelihood */
  double lpost_bl;           /* regression (each i) log posterior probability */
  double lpost_map;          /* best total log posterior probability */
  int which_map;             /* the time index giving the MAP sample */
  double llik_bl;            /* regression (each i) log likelihood */
  double llik_norm_bl;       /* as above, strictly under Normal errors for BF */
  
  /* utility vectors for addy, used for all i */
  double *s21;               /* utility vector for calcing S from beta */
  double *yvec;              /* for (temporarily) storing columns of Y */
  
  /* for printing traces */
  FILE *trace_mu;            /* traces of the mean */
  FILE *trace_S;             /* traces of the covariance */
  FILE **trace_lasso;        /* traces of the individual regressions */
  FILE *trace_DA;            /* traces of the inputs sampled by data augmentation */

  /* for collecting means and variances -- allocated externally */
  MVNsum *mom1;              /* retains the sum of mu and S samples */
  double *lambda2_sum;       /* retains the sum of lambda2 samples */
  double *m_sum;             /* retains the sum of m samples */
  MVNsum *mom2;              /* retains the sum of mu^2 and S^2 samples */
  MVNsum *mu_mom;            /* retains the sum of mu and sum(mu_i * mu_j) */
  MVNsum *map;               /* Maximum a' Posteriori mu and S */
  MVNsum *nzS;               /* retains the sum of sample indicators S != 0 */
  MVNsum *nzSi;              /* retains the sum of sample indicators inv(S) != 0 */

  /* for collecting samples of w from via Quadratic Programming */
  QPsamp *qps;               /* as above */

 protected:

  double Draw(const double thin, const bool economize, const bool burnin,
	      double *llik, double *llik_norm);
  void DataAugment(unsigned int col, const double mu, double *beta, 
		   const double s2, const double nu);

 public:

  /* constructors and destructors */
  Bmonomvn(const unsigned int M, const unsigned int N, double **Y, int *n,
	   Rmiss *R, const double p, const double theta, const unsigned int verb, 
	   const bool trace);
  ~Bmonomvn();

  /* Initialization */
  void InitBlassos(const unsigned int method, int *facts, const unsigned int RJm, 
		   const bool capm, double *mu_start, double ** S_start, 
		   int *ncomp_start, double *lambda_start, double *mprior,
		   const double r, const double delta, const bool rao_s2, 
		   const bool economy, const bool trace);

  /* sampling from the posterior distribution */
  void Rounds(const unsigned int T, const double thin, const bool economy, 
	      const bool burnin, double *nu, double *llik, double *llik_norm);

  /* printing and tracing */
  double LpostMAP(int *which);
  void InitBlassoTrace(unsigned int m);
  void InitBlassoTrace(const bool trace);
  void PrintTrace(unsigned int m);
  void Methods(int *methods);
  void Thin(const double thin, int *thin_out);
  int Verb(void);

  /* setting pointers to allocated memory */
  void SetSums(MVNsum *mom1, MVNsum *mom2, double *lambda2_sum, double *m_sum, 
	       MVNsum *map, MVNsum *mu_mom, MVNsum *nzS, MVNsum *nzSi);
  void SetQPsamp(QPsamp *qps);
};


/* getting the regression coefficients from a mean vector and covariance matrix */
void get_regress(const unsigned int m, double *mu, double *s21, double **s11, 
		 const unsigned int ncomp, double *mu_out, double *beta_out, 
		 double *s2_out);
#endif
