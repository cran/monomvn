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


/* 
 * This file contains Utility Structures for bmonomvn.cc/.h and
 * blasso.cc/.h 
 */


#ifndef __USTRUCTS_H__
#define __USTRUCTS_H__ 

#include "matrix.h"

/*
 * structure for holding the [i][j] indices
 * of entries of a m*n matrix that are missing
 * and would requre imputation by data augmentation
 * in order for ML or Bayesian inference to be
 * performed via monomvn
 */

typedef struct Rmiss
{
  unsigned int m;           /* number of cols in R (cols in X) */
  unsigned int n;           /* number ofrows in R (rows in X) */
  int **R;                  /* n * m matrix with missingness pattern */
  unsigned int *n2;         /* length of each row i of R2 (missing in X[,i]) */
  unsigned int  **R2;       /* pointers to integer lists of length n[i] */
} Rmiss;


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


/*
 * Quadratic Programming specification structure
 * and storage for samples from the posterior
 * distribution of the solution
 */

typedef struct QPsamp
{
  unsigned int m;           /* number of weights/rows in Amat/cols in W */
  unsigned int T;           /* total number of samples/rows in W */
  int *cols;                /* columns of mu/S that are not for factors */
  double **S_copy;          /* copy of S for qpgen2 */
  double *dvec;             /* vector of length m, either mu-samps or zeros */
  double *dvec_copy;        /* copy of dvec for qpgen2 */
  bool dmu;                 /* indicates if d is be replaced with mu-samps */
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
		     int *nf, double *dvec, const bool dmu, 
		     double *Amat, double *b0, int *mu_constr, 
		     const unsigned int q, const unsigned int meq, double *w);
void delete_QPsamp_R(QPsamp *qps);
void QPsolve(QPsamp *qps, const unsigned int t, const unsigned int m,
	     double *mu, double **S);

extern "C"{
#define qpgen2 qpgen2_
extern void qpgen2(double*, double*, int*, int*, double*, double*, double*,
		   double*, int*, int*, int*, int*, int*, int*, double*, 
		   int*);
}

/* functions specifically to deal with the missingness matrix, R */
void mean_of_each_col_miss(double *mean, double **M, unsigned int *n1, 
		       unsigned int n2, Rmiss *R);
void sum_of_each_col_miss_f(double *s, double **M, unsigned int *n1, 
			  unsigned int n2, Rmiss *R, double(*f)(double));
void print_Rmiss(Rmiss *R, FILE *outfile, const bool tidy);
void delete_Rmiss_R(Rmiss *R);
Rmiss* new_Rmiss_R(int *R_in, const unsigned int n, const unsigned int m);
void print_Rmiss_Xhead(Rmiss *R, FILE *outfile);
void print_Rmiss_X(Rmiss *R, double **X, const unsigned int n, 
		   const unsigned int m, FILE *outfile, PRINT_PREC type);

#endif
