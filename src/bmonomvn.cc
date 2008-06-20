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


/*
 * Bmonomvn:
 *
 * the typical constructor function for a module
 * which allows sampling from the posterior 
 * distribution with a monomvn likelihood and priors
 * with the parameteres provided
 */

Bmonomvn::Bmonomvn(const unsigned int M, const unsigned int N, double **Y, 
		   int *n, const double p, const unsigned int verb, 
		   const bool trace)
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

  /* initialize the summary vectors to NULL */
  lambda2_sum = m_sum = mu_sum = mu2_sum = NULL;
  S_sum = S2_sum = NULL;

  /* allocate blasso array */
  blasso = (Blasso**) malloc(sizeof(Blasso*) * M);

  /* utility vectors for addy */
  beta = new_zero_vector(M);
  tau2i = new_zero_vector(M);
  s21 = new_zero_vector(M);
  y = new_vector(N);
  s2 = 1.0;

  /* initialize trace files to NULL */
  trace_mu = trace_S = NULL;
  trace_lasso = NULL;
   
  /* open the trace files */
  if(trace) {
    trace_mu = fopen("mu.trace", "w");
    trace_S = fopen("S.trace", "w");
  } 

  /* need to call InitBlassos immediately */
  for(unsigned int i=0; i<M; i++) blasso[i] = NULL;
}



/*
 * ~Bmonomvn:
 *
 * the usual destructor function 
 */

Bmonomvn::~Bmonomvn(void)
{
  if(mu) free(mu);
  if(S) delete_matrix(S);

  /* clean up Bayesian Lasso regressions */
  if(blasso) {
    for(unsigned int i=0; i<M; i++) 
      if(blasso[i]) delete blasso[i];
    free(blasso);
  }

  /* clean up utility storage */
  if(beta) free(beta);
  if(tau2i) free(tau2i);
  if(s21) free(s21);
  if(y) free(y); 

  /* clean up traces */
  if(trace_lasso) {
    fclose(trace_mu);
    fclose(trace_S);
    for(unsigned int i=0; i<M; i++) 
      fclose(trace_lasso[i]);
    free(trace_lasso);
  }
}


/*
 * InitBlassos:
 *
 * initialize the Bayesian Lasso regressions used
 * as the main workhorse of the Bayesian monomvn
 * algorithm -- don't want to do any mallocing inside
 * this function since it is possible that one of the
 * "new Blasso()" calls will fail do to an invalid
 * regression, and result in an error() call that 
 * passes contrl back to R for cleanup
 */

void Bmonomvn::InitBlassos(const unsigned int method, const bool capm, 
			   double *mu_start, double **S_start, int *ncomp_start,
			   double *lambda_start, const double r, 
			   const double delta, const bool rao_s2, const bool trace)
{
  /* initialize each of M regressions */
  for(unsigned int i=0; i<M; i++) {
    
    /* get the j-th column */
    for(unsigned int j=0; j<(unsigned)n[i]; j++) y[j] = Y[j][i];

    /* choose baseline regression model to be modified by method */
    REG_MODEL rm = LASSO;
    bool RJ = true;
    if(p*((double)n[i]) > i) { RJ = FALSE; rm = OLS; }

    /* rjlsr: uses only lsr with RJ except when big-p-small-n */
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

    /* see if we can get a starting beta and s2 value */
    double *beta_start;
    double lambda2;
    if(mu_start) {
      assert(S_start && ncomp_start && lambda_start);
      get_regress(i, mu_start, S_start[i], S_start, ncomp_start[i], &mu_s, beta, &s2);
      beta_start = beta;
      lambda2 = sq(lambda_start[i]) / (4.0 * s2);
      assert(lambda2 >= 0);
    } else { 
      beta_start = NULL;
      lambda2 = (double) (rm == LASSO);
    }

    /* set up the j-th regression, with initial params */
    blasso[i] = new Blasso(i, n[i], Y, y, RJ, mmax, beta_start, s2,
			   lambda2, r, delta, rm, rao_s2, verb-1);
    blasso[i]->Init();
  }

  /* initialize traces if we're gathering them */
  if(trace) {
    trace_lasso = (FILE**) malloc(sizeof(FILE*) * M);
    for(unsigned int i=0; i<M; i++) {
      trace_lasso[i] = NULL;
      InitTrace(i);
    }
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
  assert(blasso[i]);

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
  /* sanity check */
  if(!burnin) assert(mu_sum && mu2_sum && S_sum && S2_sum);

  /* for helping with periodic interrupts */
  time_t itime = time(NULL);

  for(int t=0; t<(int)T; t++) {
    
    /* progress meter */
    if(verb && (t>0) && (t<=(int)T-1) && ((t+1) % 100 == 0)) 
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
      for(unsigned int i=0; i<M; i++)
	for(unsigned int j=0; j<M; j++) S[i][j] *= S[i][j];
      add_matrix(1.0, S2_sum, 1.0, S, M, M);
    }

    /* periodically check R for interrupts and flush console every second */
    itime = my_r_process_events(itime);
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
  /* sanity checks */
  if(!burnin) assert(lambda2_sum && m_sum);

  /* for each column of Y */
  for(unsigned int i=0; i<M; i++) {
    
    /* obtain a draw from the i-th Bayesian lasso parameters */
    blasso[i]->Draw(thin, &lambda2, &mu_s, beta, &m, &s2, tau2i, &lpost);

    /* nothing more to do when burning in */
    if(burnin) continue;

    /* possibly add a line to the trace file */
    if(trace_lasso) PrintTrace(i);

    /* add to lambda and m sums in the i-th position */
    lambda2_sum[i] += lambda2;
    m_sum[i] += m;
      
    /* update next component of the mean vector */
    mu[i] = mu_s;
    if(i > 0) mu[i] += linalg_ddot(i, beta, 1, this->mu, 1);

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


/*
 * Methods:
 *
 * get the regression method associated with each regression
 * model encoded as an integer 
 */

void Bmonomvn::Methods(int *methods)
{
  assert(methods);
  for(unsigned int i=0; i<M; i++)
    methods[i] = blasso[i]->Method();
}


/*
 * Verb:
 *
 * return the verbosity argument 
 */

int Bmonomvn::Verb(void)
{
  return verb;
}


/*
 * SetSums:
 *
 * set the pointers to the sums of samples from the
 * monomvn parameters to the pointers to memory allocated
 * outside the module 
 */

void Bmonomvn::SetSums(double *mu_sum, double *mu2_sum, double **S_sum, 
		       double **S2_sum, double *lambda2_sum, double *m_sum)
{
  this->mu_sum = mu_sum;
  this->mu2_sum = mu2_sum;
  this->S_sum = S_sum;
  this->S2_sum = S2_sum;
  this->lambda2_sum = lambda2_sum;
  this->m_sum = m_sum;
}


/*
 * get_regress:
 *
 * take a mean vector mu and covariance matrix S, with
 * components (columns) sorted according to a monotone ordering
 * and reconstruct the resulting intercepts, regression vectors
 * and variances that would have been used by the monomvn
 * algorithm -- to extract the m-th components
 */

void get_regress(const unsigned int m, double *mu, double *s21, double **s11, 
		 const unsigned int ncomp, double *mu_out, double *beta_out, 
		 double *s2_out)
{
  /* special case for first component */
  if(m == 0) {
    *mu_out = mu[0];
    *s2_out = s21[0];
    return;
  }

  /* when m >= 1 */

  /* could save on some comutation by doing this sequentially */
  /* first calculate Si = inv(S) */
  double ** s11util = new_dup_matrix(s11, m, m);
  double ** s11i = new_id_matrix(m); 
  int info = linalg_dposv(m, s11util, s11i);
  assert(info == 0);

  /* beta <- drop(s21 %*% solve(s11)) */
  linalg_dsymv(m, 1.0, s11i, m, s21, 1, 0.0, beta_out, 1);

  /* s2 <- drop(s22 - s21 %*% beta) */
  /* do this before zeroing -- perhaps move back later after 
     monomvn has been changed to handle each column individually */
  *s2_out = s21[m] - linalg_ddot(m, s21, 1, beta_out, 1);

  /* do not have the analog of the na-checking here for grouped
     lasso regress like there is in the R version of this function */

  /* for parsimony -- and so that the ncomps line up since
     the monomvn lasso version does multiple regressions 
     for particular components */
  if(ncomp < m) {
    double *beta_abs = new_vector(m);
    for(unsigned int i=0; i<m; i++) beta_abs[i] = fabs(beta_out[i]);
    double kmin = quick_select(beta_abs, m, m-ncomp-1);
    /* quick select changes the order of the entries */
    if(beta_abs) free(beta_abs);
    for(unsigned int i=0; i<m; i++)  
      if(fabs(beta_out[i]) <= kmin) beta_out[i] = 0.0;
  }

  /* for parsimony */
  /* for(unsigned int i=0; i<m-1; i++)
     if(fabs(beta_out[i]) < sqrt(DOUBLE_EPS)) beta_out[i] = 0.0; */

  /* b0 <- drop(m2 - beta %*% m1) */
  *mu_out = mu[m] - linalg_ddot(m, beta_out, 1, mu, 1);

  /* clean up */
  delete_matrix(s11util);
  delete_matrix(s11i);
}


extern "C"
{
/*
 * bmonomvn_R
 *
 * function currently used for testing the above functions
 * using R input and output
 */

double **Y = NULL;
Bmonomvn *bmonomvn = NULL;

/* starting matrices */
double **S_start = NULL;
double **S_mean = NULL;
double **S_var = NULL;

void bmonomvn_R(int *B, int *T, int *thin, int *M, int *N, double *Y_in, 
		int *n,	double *p, int *method, int *capm, double *mu_start, 
		double *S_start_in, int *ncomp_start, double *lambda_start,
		double *r, double *delta, int *rao_s2, int *verb, int *trace, 
		double *mu_mean, double *mu_var, double *S_mean_out, 
		double *S_var_out, int *methods, double *lambda2_mean, 
		double *m_mean)
{
  /* copy the vector input Y into matrix form */
  Y = new_matrix_bones(Y_in, *N, *M);

  /* copy the vector input S_start into matrix form */
  if(S_start_in) S_start = new_matrix_bones(S_start_in, *M, *M);

  /* copy the vectors S_out S_var_out into matrix form */
  S_mean = new_matrix_bones(S_mean_out, *M, *M);
  S_var = new_matrix_bones(S_var_out, *M, *M);

  /* get the random number generator state from R */
  GetRNGstate();

  /* create a new Bmonomvn object */
  bmonomvn = new Bmonomvn(*M, *N, Y, n, *p, *verb, (bool) (*trace));

  /* PERHAPS CONSIDER MOVING THIS CALL TO OUTSIDE THIS CONSTRUCTOR */
  bmonomvn->InitBlassos(*method, (bool) (*capm), mu_start, S_start, ncomp_start, 
			lambda_start, *r, *delta, (bool) (*rao_s2), 
			(bool) (*trace));

  /* do burn-in rounds */
  if(*verb) myprintf(stdout, "%d burnin rounds\n", *B);
  bmonomvn->Rounds(*B, *thin, true);
  
  /* set up the mu and S sums for calculating means and variances */
  bmonomvn->SetSums(mu_mean, mu_var, S_mean, S_var, lambda2_mean, m_mean);

  /* and now sampling rounds */
  if(*verb) myprintf(stdout, "%d sampling rounds\n", *T);
  bmonomvn->Rounds(*T, *thin, false);

  /* copy back the mean and variance of mu */
  scalev(mu_mean, *M, 1.0/(*T));
  scalev(mu_var, *M, 1.0/(*T));
  for(unsigned int i=0; i<(unsigned) *M; i++) mu_var[i] -= sq(mu_mean[i]);

  /* copy back the mean and variance of S */
  scalev(*S_mean, (*M)*(*M), 1.0/(*T));
  scalev(*S_var, (*M)*(*M), 1.0/(*T));
  for(unsigned int i=0; i<(unsigned) (*M); i++) 
    for(unsigned int j=0; j<(unsigned) (*M); j++)
      S_var[i][j] -= sq(S_mean[i][j]);

  /* bopy back the mean lambda2s */
  scalev(lambda2_mean, *M, 1.0/(*T));

  /* bopy back the mean ms */
  scalev(m_mean, *M, 1.0/(*T));

  /* get the actual methods used for each regression */
  bmonomvn->Methods(methods);

  /* clean up */
  delete bmonomvn;
  bmonomvn = NULL;

  /* give the random number generator state back to R */
  PutRNGstate();

  /* clean up */
  free(Y); Y = NULL;
  free(S_start); S_start = NULL;
  free(S_mean); S_mean = NULL;
  free(S_var); S_var = NULL;
 }


/*
 * bmonomvn_cleanup
 *
 * function for freeing memory when bmonomvn is interrupted
 * by R, so that there won't be a (big) memory leak.  It frees
 * the major chunks of memory, but does not guarentee to 
 * free up everything
 */

void bmonomvn_cleanup(void)
{
  /* free bmonomvn model */
  if(bmonomvn) { 
    if(bmonomvn->Verb() >= 1)
      myprintf(stderr, "INTERRUPT: bmonomvn model leaked, is now destroyed\n");
    delete bmonomvn; 
    bmonomvn = NULL; 
  }

  /* clean up matrix pointers */
  if(Y) { free(Y); Y = NULL; }
  if(S_start) { free(S_start); S_start = NULL; }
  if(S_mean){ free(S_mean); S_mean = NULL; }
  if(S_var){ free(S_var); S_var = NULL; }
}


/*
 * get_regress_R:
 *
 * R interface helper function for get_regress() above
 * to extract the m-th components 
 */

void get_regress_R(int *M, int *m, double *mu, double *S_in, int *ncomp,
		   double *mu_out, double *beta_out, double *s2_out)
{
  assert(*m >= 1);

  /* create a proper double ** pointer to the covarance matrix */
  double **S = (double **)  malloc(sizeof(double*) * (*M));
  S[0] = S_in;
  for(int i=1; i<(*M); i++) S[i] = S[i-1] + (*M);  

  /* call the main function to extract the m-th components */
  get_regress((*m)-1, mu, S[(*m)-1], S, *ncomp, mu_out, beta_out, s2_out);

  /* clean up */
  free(S);
}
}
