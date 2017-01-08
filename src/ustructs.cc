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

#include "Rmath.h"
#include "R.h"

extern "C"
{
#include "rhelp.h"
#include "matrix.h"
#include "linalg.h"
#include <assert.h>
}
#include "ustructs.h"


/*
 * mean_of_each_col_miss:
 *
 * fill mean[n1] with the weighted mean of the columns of M (n1 x n2);
 * using only those entries of M which have R=0
 */

void mean_of_each_col_miss(double *mean, double **M, unsigned int *n1, 
			   unsigned int n2, Rmiss *R)
{
  unsigned int i,j, count;
 
  /* sanity checks */
  if(n2 <= 0) {return;}
  assert(mean && M);
  
  /* calculate mean of columns */
  for(i=0; i<n2; i++) {
    mean[i] = 0;
    count = 0;
    for(j=0; j<n1[i]; j++) {
      if(R && R->R[j][i] != 0) continue;
      mean[i] += M[j][i];	
      count++;
    }
    mean[i] = mean[i]/count;
  }
}


/*
 * sum_of_each_col_miss_f:
 *
 * fill sum[n1] with the sum of the columns of M (n1[i] x n2);
 * each element of which is sent through function f() first;
 * n1 must have n2 entries, using only those entries of M which
 * have R=0
 */

void sum_of_each_col_miss_f(double *s, double **M, unsigned int *n1, 
			    unsigned int n2, Rmiss *R, double(*f)(double))
{
  unsigned int i,j;

  /* sanity checks */
  if(n2 <= 0) {return;}
  assert(s && M);
  
  /* calculate mean of columns */
  for(i=0; i<n2; i++) {
    s[i] = 0;
    for(j=0; j< (unsigned int) n1[i]; j++) {
      if(R && R->R[j][i] != 0) continue;
      s[i] += f(M[j][i]);
    }
  }
}



/*
 * new_MVNsum_R:
 *
 * take pointers (allocated in R) and collect them into a
 * MVNsum object for tallying a mean or variance for the
 * MVN mean vector and covariance matrix
 */

MVNsum* new_MVNsum_R(const unsigned int m, double* mu, double* S)
{
  MVNsum* mvnsum = (MVNsum*) malloc(sizeof(struct MVNsum));
  mvnsum->m = m;
  mvnsum->T = 0;
  mvnsum->mu = mu;
  mvnsum->S = new_matrix_bones(S, m, m);
  return(mvnsum);
}


/*
 * delete_MVNsum_R:
 *
 * destroy the MVNsum structure that had most
 * of its memory allocated in R
 */

void delete_MVNsum_R(MVNsum *mvnsum)
{
  free(mvnsum->S);
  free(mvnsum);
}


/*
 * MVN_add:
 *
 * add in the mu and S mean -- the covariance paramerers -- 
 * into the structure accumulating the first moment
 */

void MVN_add(MVNsum *mom1, double *mu, double **S, const unsigned int m)
{
  /* sanity check */
  assert(mom1->m == m);
  assert(mom1->mu != NULL);
  assert(mom1->S != NULL);

  /* add in the mean and covariance */
  add_vector(1.0, mom1->mu, 1.0, mu, m);
  add_matrix(1.0, mom1->S, 1.0, S, m, m);
  
  /* increment the counter of the number of things in the sum */
  (mom1->T)++;
}


/*
 * MVN_add:
 *
 * add in the product of components of mu
 * into the structure accumulating the first moment
 * and the sum of the product of the components of mu
 */

void MVN_add(MVNsum *mu_mom, double *mu, const unsigned int m)
{
  /* sanity check */
  assert(mu_mom->m == m);
  assert(mu_mom->mu == NULL);
  assert(mu_mom->S != NULL);

  /* add in the mean and covariance */
  for(unsigned int i=0; i<m; i++)
    for(unsigned int j=0; j<m; j++)
      mu_mom->S[i][j] += mu[i]*mu[j];
  
  /* increment the counter of the number of things in the sum */
  (mu_mom->T)++;
}


/*
 * MVN_add_nzS:
 *
 * add in the indicator that and S and inv(S) -- the cov 
 * paramerers and inverse --  are nonzero into the structure 
 */

void MVN_add_nzS(MVNsum *nzS, MVNsum *nzSi, double **S, const unsigned int m)
{
  /* sanity check */
  assert(nzS->m == m);
  assert(nzS->mu == NULL);
  assert(nzS->S != NULL);

  /* add in the and covariance non-zeros */
  for(unsigned int i=0; i<m; i++) 
    for(unsigned int j=0; j<m; j++)
      nzS->S[i][j] += S[i][j] != 0;

  /* invert S and then add in the inv-coariance non-zeros */
  /* Si = inv(S) */
  double **Schol = new_dup_matrix(S, m, m);
  double **Si = new_id_matrix(m);
  linalg_dposv(m, Schol, Si);
  /* now Schol = chol(S) */  
  delete_matrix(Schol);

  /* add in the and covariance non-zeros */
  for(unsigned int i=0; i<m; i++) {
    nzSi->S[i][i] += 1.0;
    for(unsigned int j=i+1; j<m; j++) {
      nzSi->S[j][i] += Si[j][i] != 0;
      nzSi->S[i][j] = nzSi->S[j][i];
    }
  }
  delete_matrix(Si);
  
  /* increment the counter of the number of things in the sum */
  (nzS->T)++;
  (nzSi->T)++;
}


/*
 * MVN_add:
 *
 * add in the square of mu and S -- the mean and covariance params --
 * into the structure accumulating the second moment
 *
 * COULD make this function omre general by adding a function pointer
 * to the sq function to allow for other functions than sq 
  */

void MVN_add2(MVNsum *mom2, double *mu, double **S, const unsigned int m)
{
  /* sanity check */
  assert(mom2->m == m);
  assert(mom2->mu != NULL);

  /* add in the square of the mean */
  for(unsigned int i=0; i<m; i++) 
    mom2->mu[i] += sq(mu[i]);

  /* add in the square of covariance */
  for(unsigned int i=0; i<m; i++)
    for(unsigned int j=0; j<m; j++) 
      mom2->S[i][j] += sq(S[i][j]);

  /* increment the counter of the number of things in the sum */
  (mom2->T)++;
}


/*
 * MVN_mean:
 *
 * calculate the mean of mu and S -- the mean and covariance params --
 * from the first moment by dividing by T and then resetting T;
 * result is in mom1 
 */

void MVN_mean(MVNsum *mom1, const unsigned int T)
{
  /* sanity check */
  assert(mom1->T == T);

  /* divite by T */
  if(mom1->mu != NULL) scalev(mom1->mu, mom1->m, 1.0/T);
  scalev(*(mom1->S), (mom1->m)*(mom1->m), 1.0/T);

  /* reset T */
  mom1->T = 0;
}


/* 
 * MVN_var:
 * valculate the variance of mu and S -- the mean and covariance params --
 * from the mean and second moment, then reset T; result is in mom2
 */

void MVN_var(MVNsum *mom2, MVNsum *mean, const unsigned int T)
{
  /* sanity check */
  assert(mom2->T == T);
  assert(mean->T == 0);

  /* calculate the variance of the mean */
  scalev(mom2->mu, mom2->m, 1.0/T);
  for(unsigned int i=0; i< mom2->m; i++) 
    mom2->mu[i] -= sq(mean->mu[i]);

  /* calculate the variance of the covariance matrix */
  scalev(*(mom2->S), (mom2->m)*(mom2->m), 1.0/T);
  for(unsigned int i=0; i<mom2->m; i++) 
    for(unsigned int j=0; j<mom2->m; j++)
      mom2->S[i][j] -= sq(mean->S[i][j]);

  /* reset T */
  mom2->T = 0;
}


/*
 * MVN_mom2cov:
 *
 * convert the mean of E[mu_i * mu_j] into Cov(mu_i, mu_j)
 */

void MVN_mom2cov(MVNsum *cov, MVNsum *mean)
{
  /* sanity check */
  assert(cov->m == mean->m);
  assert(cov->mu == NULL);
  assert(mean->mu != NULL);
  assert(cov->S != NULL);

  /* subtract off the product of means from the mean of products */
  for(unsigned int i=0; i<cov->m; i++)
    for(unsigned int j=0; j<cov->m; j++)
      cov->S[i][j] -= mean->mu[i] * mean->mu[j];
}


/*
 * MVN_copy:
 *
 * copy in the mu and S -- the covariance paramerers -- 
 * into the structure (for the maximum a posteriori)
 */

void MVN_copy(MVNsum *map, double *mu, double **S, const unsigned int m)
{
  /* sanity check */
  assert(map->m == m);

  /* add in the mean and covariance */
  dupv(map->mu, mu, m);
  dupv(*(map->S), *S, m*m);
  
  /* record that these vectors and matricies are not empty */
  map->T = 1;
}


/*
 * new_QPsamp_R:
 *
 * allocate the structure used to store the Quadratic Programmig
 * inputs and the space for samples from the solution(s) -- 
 * everything allocated in R
 */

QPsamp* new_QPsamp_R(const unsigned int m, const unsigned int T, 
		     int *cols, double *dvec,  const bool dmu, 
		     double *Amat, double *b0, int *mu_constr, 
		     const unsigned int q, const unsigned int meq, double *w)
{
  /* check if QP is required */
  if(w == NULL) return NULL;

  /* sanity check */
  assert(q > 0 && m > 0);

  /* allocate the QPsamp structure */
  QPsamp* qps = (QPsamp*) malloc(sizeof(struct QPsamp));

  /* record dimensional parameters */
  qps->m = m;
  qps->T = T;

  /* record the factor labels */
  qps->cols = new_dup_ivector(cols, m);
  
  /* make space to copy S sampls into */
  qps->S_copy = new_matrix(m, m);

  /* copy dvec (which could be mu-samples) */
  qps->dvec = dvec;
  qps->dvec_copy = new_vector(m);
  qps->dmu = dmu;

  /* copy the linear constraints */
  qps->q = q;
  qps->Amat = Amat;
  qps->b0 = b0;
  qps->meq = meq;

  /* copy the info on mu in constraints */
  qps->mu_c_len = (unsigned int) *mu_constr;
  if(qps->mu_c_len == 0) {
    qps->mu_c = NULL;
    qps->Amat_copy = qps->Amat;
    qps->b0_copy = qps->b0;
  } else {
    qps->mu_c = mu_constr + 1;
    qps->Amat_copy = new_vector(q*m);
    qps->b0_copy = new_vector(q);
  }

  /* allocate workspace */
  unsigned int r = q;
  if(m < q) r = m;
  qps->iact = new_zero_ivector(q);
  qps->work = new_zero_vector(2*m + r*(r+5)/2 + 2*q + 1);

  /* turn the W (or B) samples of w (or b) into a matrix */
  qps->W = new_matrix_bones(w, T, m);

  /* return the new structure */
  return qps;
}


/*
 * delete_QPsamp_R:
 *
 * destroy the QPsamp structure that had most
 * of its memory allocated in R
 */

void delete_QPsamp_R(QPsamp *qps)
{
  delete_matrix(qps->S_copy);
  free(qps->dvec_copy);
  if(qps->mu_c_len > 0) {
    free(qps->Amat_copy);
    free(qps->b0_copy);
  }
  free(qps->cols);
  free(qps->iact);
  free(qps->work);
  free(qps->W);
  free(qps);
}


/*
 * QPsolve:
 *
 * solve the Quadric Program with the current value
 * of mu and S and store the result in W[t], thus
 * taking the t-th sample of the solution w (or b)
 */

void QPsolve(QPsamp *qps, const unsigned int t, const unsigned int m,
	     double *mu, double **S)
{
  int ierr, nact;
  int iter[2];
  double crval;

  /* sanity checks */
  assert(qps);
  assert(qps->m <= m);
  assert(t < qps->T);

  /* check to see if we should fill in dvec with the mean */
  dupv(qps->dvec_copy, qps->dvec, qps->m);
  if(qps->dmu) {
    for(unsigned int i=0; i<qps->m; i++)
      qps->dvec_copy[i] *= mu[qps->cols[i]];
  }

  /* copy S */
  if(m == qps->m) dupv(qps->S_copy[0], S[0], m*m);
  else { /* otherwise need to copy only certain rows and cols */
    for(unsigned int i=0; i<qps->m; i++)
      for(unsigned int j=0; j<qps->m; j++)
	qps->S_copy[i][j] = S[qps->cols[i]][qps->cols[j]];
  }

  /* copy Amat and b0 */
  if(qps->mu_c_len != 0) {
    dupv(qps->b0_copy, qps->b0, qps->q);
    dupv(qps->Amat_copy, qps->Amat, (qps->q)*qps->m);
  }

  /* check to see if we should fill in rows of Amat with mu */
  for(unsigned int i=0; i<qps->mu_c_len; i++)
    for(unsigned int j=0; j<qps->m; j++) 
      qps->Amat_copy[qps->m*(qps->mu_c[i]-1)+j] *= mu[qps->cols[j]];
  
  /* ready to solve */
  ierr = 0;  /* indicates S is not decomposed */
  qpgen2(*(qps->S_copy), qps->dvec_copy, (int*) &(qps->m), 
	 (int*) &(qps->m), qps->W[t], &crval, qps->Amat_copy, 
	 qps->b0_copy, (int*) &(qps->m), (int*) &(qps->q), 
	 (int*) &(qps->meq), qps->iact, &nact, iter, qps->work, 
	 &ierr);

  /* MIGHT make all of the W's positive, and then renormalize */
  
  /* MIGHT want to check ierr */
  // MYprintf(stdout, "ierr = %d\n", ierr);
  assert(ierr==0);
}



/*
 * new_Rmiss_R:
 *
 * take in an R matrix contatning a missingness pattern,
 * allocated in R, and convert it into a list of indices
 * to the entries in R which are 2, etc.
 */

Rmiss* new_Rmiss_R(int *R_in, const unsigned int n, const unsigned int m)
{
  /* do nothing if R_in is NULL */
  if(R_in == NULL) return NULL;

  /* allocate space for the struct and its pointers */
  Rmiss *R = (Rmiss*) malloc(sizeof(struct Rmiss));
  R->m = m;
  R->n = n;
  R->R = new_imatrix_bones(R_in, n, m);
  R->n2 = new_uivector(m);
  R->R2 = (unsigned int **) malloc(sizeof(unsigned int *) *m);
  
  /* loop over the columns of R */
  unsigned int n2_sum = 0;
  for(unsigned int i=0; i<m; i++) {

    /* count the number which have 2-entries */
    R->n2[i] = 0;
    for(unsigned int j=0; j<n; j++)
      if(R->R[j][i] == 2) (R->n2[i])++;
    n2_sum += R->n2[i];

    /* allocate the R row and fill it */
    if(R->n2[i] > 0) {
      R->R2[i] = new_uivector(R->n2[i]);

      /* fill */
      unsigned int k=0;
      for(unsigned int j=0; j<n; j++)
      if(R->R[j][i] == 2) R->R2[i][k++] = j;

    } else R->R2[i] = NULL;
  }

  /* return NULL if there is no DA needed */
  if(n2_sum == 0) { delete_Rmiss_R(R); R = NULL; }

  /* return */
  return(R);
} 


/* 
 * delete_Rmiss:
 *
 * free the Rmiss structure and its pointers
 */

void delete_Rmiss_R(Rmiss *R)
{
  for(unsigned int i=0; i<R->m; i++)
    if(R->R2[i] != NULL) free(R->R2[i]);
  free(R->n2);
  free(R->R2);
  free(R->R);
  free(R);
}


/*
 * print_Rmiss:
 *
 * print the contents of an Rmiss strucure out to
 * a file
 */

void print_Rmiss(Rmiss *R, FILE *outfile, const bool tidy)
{
  /* nothing to print if NULL */
  if(!R) { MYprintf(outfile, "Rmiss is NULL\n"); return; }

  /* print dimensions */
  MYprintf(outfile, "Rmiss: n=%d, m=%d\nR=\n", R->n, R->m);

  if(!tidy) {
    /* print the transpose matrix */
    printIMatrix(R->R, R->n, R->m, outfile);
  }

  /* print the list(s) of indices */
  for(unsigned int i=0; i<R->m; i++) {
    if(tidy && R->n2[i] == 0) continue;
    MYprintf(outfile, "R2[%d] =", i, R->n2[i]);
    for(unsigned int j=0; j<R->n2[i]; j++)
      MYprintf(outfile, " %d", R->R2[i][j]);
    MYprintf(outfile, "; (%d)\n", R->n2[i]);
  }
}


/*
 * print_Rmiss_X:
 *
 * print the elements of X that correspond with R->R2
 */

void print_Rmiss_X(Rmiss *R, double **X, const unsigned int n, 
		   const unsigned int m, FILE *outfile, PRINT_PREC type)
{
  /* sanity checks */
  assert(R && outfile);
  assert(R->n == n && R->m == m);

  /* loop over the elements of R[j][i] == 2 */
  for(unsigned int i=0; i<m; i++) {
    for(unsigned int j=0; j<R->n2[i]; j++) {
      if(type==HUMAN) MYprintf(outfile, "%g ", X[R->R2[i][j]][i]);
      else if(type==MACHINE) MYprintf(outfile, "%.20f ", X[R->R2[i][j]][i]);
    }
  }
  
  /* add a newline */
  MYprintf(outfile, "\n");
}


/*
 * print_Rmiss_Xhead:
 *
 * print the table header for print_Rmiss_X
 */

void print_Rmiss_Xhead(Rmiss *R, FILE *outfile)
{
  /* sanity checks */
  assert(R && outfile);

  /* loop over the elements of R[j][i] == 2 */
  for(unsigned int i=0; i<R->m; i++) {
    for(unsigned int j=0; j<R->n2[i]; j++) {
      MYprintf(outfile, "i%dj%d ", (R->R2[i][j])+1, i+1);
    }
  }
  
  /* add a newline */
  MYprintf(outfile, "\n");
}
