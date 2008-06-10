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


extern "C"
{
#include "rhelp.h"
#include "matrix.h"
#include "linalg.h"
#include "Rmath.h"
#include "R.h"
#include "assert.h"
}
#include "bmonomvn.h"
#include "blasso.h"

Bmonomvn::Bmonomvn(const unsigned int M, const unsigned int N, double **Y, 
		   int *n, const double p, const unsigned int method, 
		   const bool capm, const double r, const double delta, 
		   const bool rao_s2, const unsigned int verb, const bool trace)
{
  /* copy inputs */
  this->M = M;
  this->N = N;
  this->n = n;
  this->Y = Y;
  this->verb = verb;
  this->p = p;

  /* allocate the mean vector and covariance matrix */
  mu = new_zero_vector(M);
  S = new_zero_matrix(M, M);

  /* allocate average mean vector and covariance matrix */
  mu_sum = new_zero_vector(M);
  mu2_sum = new_zero_vector(M);
  S_sum = new_zero_matrix(M, M);

  /* allocate the lamdba2 and m averages */
  lambda2_sum = new_zero_vector(M);
  m_sum = new_zero_vector(M);

  /* allocate blasso array */
  blasso = (Blasso**) malloc(sizeof(Blasso*) * M);

  /* initize the Bayesian Lassos */
  /* PERHAPS MAKE THIS ITS OWN FUNCTION */
  double *y = new_vector(N);
  for(unsigned int i=0; i<M; i++) {
    
    /* get the j-th column */
    for(unsigned int j=0; j<(unsigned)n[i]; j++) y[j] = Y[j][i];

    /*myprintf(stdout, "p=%g, n[%d]=%d, p*n[%d]=%g\n", 
      p, i, n[i], i, ((double)p)*n[i]);*/

    /* choose regression model */
    REG_MODEL rm = LASSO;
    bool RJ = true;

    /* rjlsr: uses only lsr wih RJ except when big-p-small-n */
    if(method == 1) {
      if(n[i] > (int) i) rm = OLS;
      else rm = LASSO;
    }

    /* rjlasso and default: involves RJ at some stage */
    if(method == 2 || (method == 3 && n[i] > (int) i)) 
      RJ = false;

    /* choose the maximum number of columns */
    unsigned int mmax = i;
    if(RJ && capm && n[i] <= (int) i) mmax = n[i]-1;

    /* set up the j-th regression, with initial params */
    if(p*((double)n[i]) > i) { /* using standard linear regression */
      blasso[i] = new Blasso(i, n[i], Y, y, false, mmax, r, 
			     delta, OLS, rao_s2, verb-1);
    } else { /* using lasso */
      blasso[i] = new Blasso(i, n[i], Y, y, RJ, mmax, r, 
			     delta, rm, rao_s2, verb-1);
    }
  }
  free(y);

  /* utility vectors for addy */
  beta = new_zero_vector(M);
  tau2i = new_zero_vector(M);
  s21 = new_zero_vector(M);

  /* open the trace files */
  if(trace) {
    trace_mu = fopen("mu.trace", "w");
    trace_S = fopen("S.trace", "w");
    trace_lasso = (FILE**) malloc(sizeof(FILE*) * M);
    for(unsigned int i=0; i<M; i++) {
      trace_lasso[i] = NULL;
      InitTrace(i);
    }
  } else {
    trace_mu = trace_S = NULL;
    trace_lasso = NULL;
  }
}


/*
 * ~Bmonomvn:
 *
 * the usual destructor function 
 */

Bmonomvn::~Bmonomvn(void)
{
  /* clean up */
  free(mu);
  free(mu_sum);
  free(mu2_sum);
  delete_matrix(S);
  delete_matrix(S_sum);
  for(unsigned int i=0; i<M; i++) delete blasso[i];
  free(blasso);
  free(beta);
  free(tau2i);
  free(s21);
  free(lambda2_sum);
  free(m_sum);
  if(trace_lasso) {
    fclose(trace_mu);
    fclose(trace_S);
    for(unsigned int i=0; i<M; i++) 
      fclose(trace_lasso[i]);
    free(trace_lasso);
  }
}


/* 
 * InitTrace:
 *
 * open the m-th trace file and and write the
 * appropriate header in the file 
 */

void Bmonomvn::InitTrace(unsigned int i)
{
  /* sanity checks */
  assert(i < M);
  assert(trace_lasso && (trace_lasso[i] == NULL));

  /* create the filename and open the file */
  char fname[256];
  sprintf(fname, "blasso_M%d_n%d.trace", i, n[i]);
  trace_lasso[i] = fopen(fname, "w");
  assert(trace_lasso[i]);

  /* add the R-type header */
  fprintf(trace_lasso[i], "s2 mu m ");
  for(unsigned int j=0; j<i; j++)
    fprintf(trace_lasso[i], "beta.%d ", j);
  
  /* maybe add lasso params to the header */
  if(blasso[i]->RegModel() == LASSO) {
    fprintf(trace_lasso[i], "lambda2 ");
    for(unsigned int j=0; j<i; j++)
      fprintf(trace_lasso[i], "tau2i.%d ", j);
  }

  /* finish off the header */
  fprintf(trace_lasso[i], "\n");
}


/*
 * PrintTrace:
 *
 * print a line to the trace file of the m-th regression
 */

void Bmonomvn::PrintTrace(unsigned int i)
{
  assert(trace_lasso && trace_lasso[i]);
  
  /* add the mean and variance to the line */
  fprintf(trace_lasso[i], "%.20f %.20f %d ", s2, mu_s, m);

  /* add the regression coeffs to the file */
  for(unsigned int j=0; j<i; j++)
    fprintf(trace_lasso[i], "%.20f ", beta[j]);

  /* maybe add lasso params to the file */
  if(blasso[i]->RegModel() == LASSO) {
    fprintf(trace_lasso[i], "%.20f ", lambda2);
    for(unsigned int j=0; j<i; j++)
      fprintf(trace_lasso[i], "%.20f ", tau2i[j]);
  }

  /* finish printing the line */
  fprintf(trace_lasso[i], "\n");
}


/*
 * PrintRegressions:
 *
 * print the input information about each regression
 * to the specified outfile
 */

void Bmonomvn::PrintRegressions(FILE *outfile)
{
  for(unsigned int i=0; i<M; i++) {
    myprintf(outfile, "regression %d\n", i);
    blasso[i]->PrintInputs(outfile);
    myprintf(outfile, "\n");
  }
}


/*
 * Rounds:
 *
 * sample from the posterior distribution of the monomvn
 * algorithm over T rounds with the thinning level provided.
 * Only record samples if burnin - FALSE.
 */

void Bmonomvn::Rounds(const unsigned int T, const unsigned int thin, 
		      const bool burnin)
{
  for(unsigned int t=0; t<T; t++) {
    
    /* progress meter */
    if(verb && (t>0) && (t<((unsigned) (T-1))) && ((t+1) % 100 == 0)) 
      myprintf(stdout, "t=%d\n", t+1);

    /* take one draw after thinning */
    Draw(thin, burnin);

    /* record samples unless burning in */
    if(! burnin) {

      /* possibly add trace samples to the files */
      if(trace_mu) printVector(mu, M, trace_mu, MACHINE);
      if(trace_S) printSymmMatrixVector(S, M, trace_S, MACHINE);

      /* add vectors and matrices for later mean calculations */
      add_vector(1.0, mu_sum, 1.0, mu, M);
      for(unsigned int i=0; i<M; i++) mu[i] *= mu[i];
      add_vector(1.0, mu2_sum, 1.0, mu, M);
      add_matrix(1.0, S_sum, 1.0, S, M, M);
    }
  }
}


/*
 * Draw:
 *
 * Take one draw from the posterior distribution of the
 * monomvn algorithm, at the thinning level provided.
 * If burnin = true then only the lasso samples are drawn,
 * and the other calculations are skipped.
 */

void Bmonomvn::Draw(const unsigned int thin, const bool burnin)
{
  /* for each column of Y */
  for(unsigned int i=0; i<M; i++) {
    
    blasso[i]->Draw(thin, &lambda2, &mu_s, beta, &m, &s2, tau2i, &lpost);

    /* nothing more to do when burning in */
    if(burnin) continue;

    /* possibly add a line to the trace file */
    if(trace_lasso) PrintTrace(i);

    lambda2_sum[i] += lambda2;
    m_sum[i] += m;

    /* update next component of the mean vector */
    this->mu[i] = mu_s;
    if(i > 0) {
      /*myprintf(stdout, "beta = ");
	printVector(beta, i, stdout, HUMAN);*/
      this->mu[i] += linalg_ddot(i, beta, 1, this->mu, 1);
    }

    /*myprintf(stdout, "i=%d, s2=%g, lambda2=%g, mu=%g, mu[%d]=%g\n",
      i, s2, lambda2, mu, i, this->mu[i]); */

    /* update the next column of the covariance matrix */
    if(i == 0) S[0][0] = s2;
    else {
      
      /* s21 <- b1 * s11 */
      linalg_dsymv(i, 1.0, S /*s11*/, M /*i*/, beta, 1, 0.0, s21, 1);

      /* put the next row and column on S */
      dupv(S[i], s21, i);
      for(unsigned int j=0; j<i; j++) S[j][i] = S[i][j];

      /* s22 <- s2 + s21 %*% b1 */
      S[i][i] = s2 + linalg_ddot(i, s21, 1, beta, 1);
    }
  }
}


void Bmonomvn::Methods(int *methods)
{
  assert(methods);
  for(unsigned int i=0; i<M; i++)
    methods[i] = blasso[i]->Method();
}


extern "C"
{
/*
 * bmonomvn_R
 *
 * function currently used for testing the above functions
 * using R input and output
 */

void bmonomvn_R(int *B, int *T, int *thin, int *M, int *N, double *Y_in, 
		int *n,	double *p, int *method, int *capm, double *r, 
		double *delta, int *rao_s2, int *verb, int *trace, 
		double *mu, double *mu_var, double *S, int *methods,
		double *lambda2_mean, double *m_mean)
{
  double **Y;
  int i;

  /* copy the vector input S into matrix form */
  Y = (double **)  malloc(sizeof(double*) * (*N));
  Y[0] = Y_in;
  for(i=1; i<(*N); i++) Y[i] = Y[i-1] + (*M);

  /* get the random number generator state from R */
  GetRNGstate();

  Bmonomvn *bmonomvn = new Bmonomvn(*M, *N, Y, n, *p, *method, (bool) (*capm), 
				    *r, *delta, (bool) (*rao_s2), *verb, 
				    (bool) (*trace));

  // bmonomvn->PrintRegressions(stdout);
  if(*verb) myprintf(stdout, "%d burnin rounds\n", *B);
  bmonomvn->Rounds(*B, *thin, true);
  if(*verb) myprintf(stdout, "%d sampling rounds\n", *T);
  bmonomvn->Rounds(*T, *thin, false);

  /* copy back the mean of mu */
  dupv(mu, bmonomvn->mu_sum, *M);
  scalev(mu, *M, 1.0/(*T));

  /* copy back the var of mu */
  dupv(mu_var, bmonomvn->mu2_sum, *M);
  scalev(mu_var, *M, 1.0/(*T));
  for(unsigned int i=0; i<(unsigned) *M; i++) mu_var[i] -= sq(mu[i]);

  /* copy back the mean of S */
  dupv(S, *(bmonomvn->S_sum), (*M)*(*M));
  scalev(S, (*M)*(*M), 1.0/(*T));

  /* bopy back the mean lambda2s */
  dupv(lambda2_mean, bmonomvn->lambda2_sum, *M);
  scalev(lambda2_mean, *M, 1.0/(*T));

  /* bopy back the mean ms */
  dupv(m_mean, bmonomvn->m_sum, *M);
  scalev(m_mean, *M, 1.0/(*T));

  /* get the actual methods used for each regression */
  bmonomvn->Methods(methods);

  delete bmonomvn;

  /* give the random number generator state back to R */
  PutRNGstate();

  /* clean up */
  free(Y);
 }
}
