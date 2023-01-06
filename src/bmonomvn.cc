/**************************************************************************** 
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
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  
 * 02110-1301  USA
 *
 * Questions? Contact Robert B. Gramacy (bobby@statslab.cam.ac.uk)
 *
 ****************************************************************************/

#include "Rmath.h"
#include "R.h"

extern "C"
{
#include "rhelp.h"
#include "matrix.h"
#include "linalg.h"
#include "nu.h"
#include <assert.h>
}
#include "ustructs.h"
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
		   int *n, Rmiss *R, const double p, const double theta, 
		   const unsigned int verb, const bool trace)
{
  /* copy inputs */
  this->M = M;
  this->N = N;
  this->n = n;
  this->Y = Y;
  this->R = R;
  // this->n2 = n2;
  this->verb = verb;
  this->p = p;

  /* calculate the mean of each column of X where not missing */
  Xmean = new_zero_vector(M);
  mean_of_each_col_miss(Xmean, this->Y, (unsigned int *) n, M, R);
  
  /* center X */
  double **X = new_dup_matrix(Y, N, M);
  center_columns(X, Xmean, N, M);
  
  /* normalize X */
  Xnorm = new_zero_vector(M);
  sum_of_each_col_miss_f(Xnorm, X, (unsigned int *) n, M, R, sq);
  for(unsigned int i=0; i<M; i++) {
    Xnorm[i] = sqrt(Xnorm[i]);
    if(R) Xnorm[i] *= sqrt(((double)n[i])/(n[i] - R->n2[i]));
  }
  /* norm_columns(X, Xnorm, N, M); */
  delete_matrix(X);
  
  /* allocate the mean vector and covariance matrix */
  mu = new_zero_vector(M);
  S = new_zero_matrix(M, M);

  /* initialize the summary vectors to NULL */
  mom1 = mom2 = map = nzS = nzSi = mu_mom = NULL;
  lambda2_sum = m_sum = NULL;

  /* initialize the QP samples to NULL */
  qps = NULL;

  /* allocate blasso array */
  blasso = (Blasso**) malloc(sizeof(Blasso*) * M);

  /* utility vectors for addy (and traces) */
  beta = new_zero_vector(M);
  s21 = new_zero_vector(M);
  yvec = new_vector(N);
  s2 = 1.0;

  /* utility doubles for traces etc */
  pi = lambda2 = gam = 0.0;
  this->theta = theta;
  onenu = false;
  if(theta != 0) {
    if(theta < 0) { onenu = true; this->theta = 0.0-theta; }
    nu = 1.0/this->theta;
  } else { nu = 0.0; onenu = false; }

  /* utility vectors for traces */
  tau2i = new_zero_vector(M);
  if(theta != 0) omega2 = new_zero_vector(N);
  else omega2 = NULL;

  /* initialize posterior probabilities and likelihoods */
  lpost_bl = lpost_map = -1e300*1e300;
  which_map = -1;
  llik_bl = llik_norm_bl = 0.0;

  /* initialize trace files to NULL */
  trace_DA = trace_mu = trace_S = NULL;
  trace_lasso = NULL;

  /* open the trace files */
  if(trace) {
    trace_mu = fopen("mu.trace", "w");
    trace_S = fopen("S.trace", "w");
    if(R) { 
      trace_DA = fopen("DA.trace", "w");
      print_Rmiss_Xhead(R, trace_DA);
    }
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
  /* copied and normalized design matrices */
  if(Xnorm) free(Xnorm);
  if(Xmean) free(Xmean);

  /* parameters */
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
  if(omega2) free(omega2);
  if(s21) free(s21);
  if(yvec) free(yvec); 

  /* clean up traces */
  if(trace_mu) fclose(trace_mu);
  if(trace_S) fclose(trace_S);
  if(trace_DA) fclose(trace_DA);
  if(trace_lasso) {
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

void Bmonomvn::InitBlassos(const unsigned int method, int* facts, 
			   const unsigned int RJm, const bool capm, 
			   double *mu_start, double **S_start,
			   int *ncomp_start, double *lambda_start, 
			   double *mprior, const double r, 
			   const double delta, const bool rao_s2, 
			   const bool economy, const bool trace)
{
  /* initialize each of M regressions */
  for(unsigned int i=0; i<M; i++) {
    
    /* get the i-th column of Y */
    for(unsigned int j=0; j<(unsigned)n[i]; j++)  yvec[j] = Y[j][i];

    /* choose baseline regression model to be modified by method & RJ */
    REG_MODEL rm = OLS;
    unsigned int nf = 0;
    if(method == 3) { 
      rm = FACTOR;
      assert(facts);
      nf = (unsigned int) p;
    }

    /* initialize RJ method */
    bool RJ = false; /* corresponds to RJm = 2 */

    /* re-set RJ and method vi the p parameter */
    if(rm != FACTOR && p*((double)n[i]) <= i+2) {  /* recently changed to i + 2 (v1.8) */
      switch (method) {
      case 0: rm = LASSO; break;
      case 1: rm = RIDGE; break;
      case 2: rm = OLS; break;
      case 3: rm = FACTOR; break;
      case 4: rm = HORSESHOE; break;
      case 5: rm = NG; break;
      default: error("regression method %d not supported", method);
      }
      
      /* now deal with RJ */
      if(RJm == 1) RJ = true; /* RJ whenever parsimonious applied */
      else if(RJm == 0 && n[i] <= (int) i) 
	RJ = true; /* RJ whenever ill-posed regression */
    }

    /* choose the maximum number of columns */
    unsigned int mmax = i;
    if(RJ && capm && n[i] <= (int) i) mmax = n[i]-1;

    /* see if we can get a starting beta and s2 value */
    double *beta_start;
    double lambda2;
    if(mu_start) {
      assert(S_start && ncomp_start && lambda_start);
      get_regress(i, mu_start, S_start[i], S_start, ncomp_start[i], 
		  &mu_s, beta, &s2);
      beta_start = beta;
      /* THIS WOULD BE INCORRECT FOR RIDGE REGRESSION */
      lambda2 = sq(lambda_start[i]) / (4.0 * s2);
      assert(lambda2 >= 0);
    } else { 
      beta_start = NULL;
      lambda2 = (double) (rm != OLS);      
    }
    
    /* starting gamma parameter (possibly) for NG prior;
       [not read from the start vector] */
    if(rm == NG) gam = 2.0;
    else gam = 1.0;

    /* set up the j-th regression, with initial params */
    double Xnorm_scale = sqrt(((double) n[i])/N);
    if(R) Xnorm_scale = sqrt(((double)(n[i] - R->n2[i]))/N);
    blasso[i] = new Blasso(i, n[i], Y, R, Xnorm, Xnorm_scale, Xmean, M, 
			   yvec, RJ, mmax, beta_start, s2, lambda2, mprior, 
			   r, delta, 1.0, theta, rm, facts, nf, rao_s2, verb-1);
    if(!economy) blasso[i]->Init();
  }

  /* initialize traces if we're gathering them */
  InitBlassoTrace(trace);
}


/*
 * InitBlassoTrace:
 *
 * initialize the traces for all of the Blasso objects
 * by looping over InitBlassoTrace(i) -- if no traces, then
 * do nothing
 */

void Bmonomvn::InitBlassoTrace(const bool trace)
{
  /* initialize traces if we're gathering them */
  if(trace) {
    trace_lasso = (FILE**) malloc(sizeof(FILE*) * M);
    for(unsigned int i=0; i<M; i++) {
      trace_lasso[i] = NULL;
      InitBlassoTrace(i);
    }
  }
}


/* 
 * InitBlassoTrace:
 *
 * open the m-th trace file and and write the
 * appropriate header in the file 
 */

void Bmonomvn::InitBlassoTrace(unsigned int i)
{
  /* sanity checks */
  assert(i < M);
  assert(trace_lasso && (trace_lasso[i] == NULL));
  assert(blasso[i]);

  /* create the filename and open the file */
  char fname[256];
  snprintf(fname, 256, "blasso_M%d_n%d.trace", i, n[i]);
  trace_lasso[i] = fopen(fname, "w");
  assert(trace_lasso[i]);

  /* add the R-type header */
  fprintf(trace_lasso[i], "llik lpost s2 mu ");
  if(blasso[i]->UsesRJ()) fprintf(trace_lasso[i], "m ");
  for(unsigned int j=0; j<i; j++)
    fprintf(trace_lasso[i], "beta.%d ", j);
  
  /* maybe add lasso params to the header */
  REG_MODEL rm = blasso[i]->RegModel();
  if(rm != OLS) {
    fprintf(trace_lasso[i], "lambda2 ");
    if(rm == NG) fprintf(trace_lasso[i], "gamma ");
    if(rm == LASSO || rm == HORSESHOE || rm == NG)
      for(unsigned int j=0; j<i; j++)
	fprintf(trace_lasso[i], "tau2i.%d ", j);
  }

  /* maybe add omega and nu to the file */
  if(blasso[i]->TErrors()) {
    fprintf(trace_lasso[i], "nu ");
    int ni = n[i];
    if(R) ni -= R->n2[i];
     for(int j=0; j<ni; j++)
	fprintf(trace_lasso[i], "omega2.%d ", j);
     fprintf(trace_lasso[i], "llik.norm ");
  }

  /* maybe add the pi parameter */
  if(!(blasso[i]->FixedPi())) fprintf(trace_lasso[i], "pi ");

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
  fprintf(trace_lasso[i], "%.20f %.20f %.20f %.20f ", 
	  llik_bl, lpost_bl, s2, mu_s);
  if(blasso[i]->UsesRJ()) fprintf(trace_lasso[i], "%d ", m);

  /* add the regression coeffs to the file */
  for(unsigned int j=0; j<i; j++)
    fprintf(trace_lasso[i], "%.20f ", beta[j]);

  /* maybe add lasso params to the file */
  REG_MODEL rm = blasso[i]->RegModel();
  if(rm != OLS) {
    fprintf(trace_lasso[i], "%.20f ", lambda2);
    if(rm == NG) fprintf(trace_lasso[i], "%.20f ", gam);
    if(rm == LASSO || rm == HORSESHOE || rm == NG)
      for(unsigned int j=0; j<i; j++)
	fprintf(trace_lasso[i], "%.20f ", tau2i[j]);
  }

  /* maybe add omega and nu to the file */
  if(blasso[i]->TErrors()) {
    fprintf(trace_lasso[i], "%.20f ", nu);
    int ni = n[i];
    if(R) ni -= R->n2[i];
     for(int j=0; j<ni; j++)
	fprintf(trace_lasso[i], "%.20f ", omega2[j]);
     fprintf(trace_lasso[i], "%.20f ", llik_norm_bl);
  }

  /* maybe add the pi param to the file */
  if(!(blasso[i]->FixedPi())) fprintf(trace_lasso[i], "%.20f ", pi);

  /* finish printing the line */
  fprintf(trace_lasso[i], "\n");
}


/*
 * Rounds:
 *
 * sample from the posterior distribution of the monomvn
 * algorithm over T rounds with the thinning level provided.
 * Only record samples if burnin - FALSE.
 */

void Bmonomvn::Rounds(const unsigned int T, const double thin, 
		      const bool economy, const bool burnin, double *nu,
		      double *llik, double *llik_norm)
{
  /* sanity check */
  if(!burnin) assert(mom1 && mom2 && nzS && nzSi && llik &&
		     map && mu_mom && m_sum && lambda2_sum);
  if(!burnin && onenu) assert(nu != NULL);

  /* for helping with periodic interrupts */
  time_t itime = time(NULL);

  for(int t=0; t<(int)T; t++) {
    
    /* progress meter */
    if(verb && (t>0) && (t<=(int)T-1) && ((t+1) % 100 == 0)) 
      MYprintf(MYstdout, "t=%d\n", t+1);

    /* take one draw after thinning */
    double llik_temp, llik_norm_temp;
    double lpost = Draw(thin, economy, burnin, &llik_temp, &llik_norm_temp);

    /* record samples unless burning in */
    if(! burnin) {

      /* possibly add trace samples to the files */
      if(trace_mu) printVector(mu, M, trace_mu, MACHINE);
      if(trace_S) printSymmMatrixVector(S, M, trace_S, MACHINE);
      if(trace_DA) print_Rmiss_X(R, Y, N, M, trace_DA, MACHINE);
      
      /* add vectors and matrices for later mean & var calculations */
      MVN_add(mom1, mu, S, M);
      MVN_add2(mom2, mu, S, M);
      MVN_add(mu_mom, mu, M);
      MVN_add_nzS(nzS, nzSi, S, M);

      /* save log likelihood */
      llik[t] = llik_temp;
      if(llik_norm) llik_norm[t] = llik_norm_temp;

      /* save a nu sample if using pooled nu */
      if(onenu) nu[t] = this->nu;

      /* check for new best MAP */
      // MYprintf(MYstdout, "lpost = %g, lpost_map = %g\n", lpost, lpost_map);
      if(lpost > lpost_map) {
        lpost_map = lpost;
	        MVN_copy(map, mu, S, M);
	        which_map = t;
      }

      /* calculate the QP solution */
      if(qps) QPsolve(qps, t, M, mu, S);
    }

    /* periodically check R for interrupts and flush console every second */
    itime = MY_r_process_events(itime);
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

double Bmonomvn::Draw(const double thin, const bool economy, 
		      const bool burnin, double *llik, double *llik_norm)
{
  /* sanity checks */
  if(!burnin) assert(lambda2_sum && m_sum);

  /* for accumulating log posterior probability */
  double lpost = 0.0;
  *llik = *llik_norm = 0.0;

  /* for accumulating eta if pooling nu*/
  double eta = theta;
  int neta = 0;

  /* for each column of Y */
  for(unsigned int i=0; i<M; i++) {

    /* need to initialse each blasso before it is used if economizing */
    if(economy) blasso[i]->Init();
    
    /* obtain a draw from the i-th Bayesian lasso parameters */
    blasso[i]->Draw(thin, onenu, &mu_s, beta, &m, &s2, tau2i, 
		    &lambda2, &gam, omega2, &nu, &pi, &lpost_bl, 
		    &llik_bl, &llik_norm_bl);

    /* accumulate eta for a global nu draw */
    if(onenu) {
      assert(theta != 0);
      int ni = n[i];
      if(R) ni -= R->n2[i];
      for (int k=0; k<ni; k++) eta += 0.5*(log(omega2[k])+1.0/omega2[k]);
      neta += ni;
    }

    /* perform data augmentation */
    DataAugment(i, mu_s, beta, s2, nu);

    /* now un-init if economizing */
    if(economy) blasso[i]->Economize();

    /* nothing more to do when burning in */
    if(burnin) continue;

    /* tally log posterior probability */
    lpost += lpost_bl;
    *llik += llik_bl;
    *llik_norm += llik_norm_bl;

    /* possibly add a line to the trace file */
    if(trace_lasso) PrintTrace(i);

    /* add to lambda and m sums in the i-th position */
    lambda2_sum[i] += lambda2;
    m_sum[i] += m;
     
    /* 
     * addy (the next component of mu and col of S) begins here
     * [perhaps make a separate function]
     */
 
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

  /* update the single, pooled eta, nu */
  if(onenu) nu = draw_nu_reject(neta, eta);  

  /* return the log posterior probability of the draw */
  return lpost;
}


/*
 * DataAugment:
 *
 * perform Data Augmentation on a particular column of the 
 * design matrix with the regression parameters provided
 *
 * CAN DO BETTER THAN THIS BY INTEGRATING OUT BETA, BY
 * USING Blasso::breg->bmu and breg->Vb, BUT BOTH WOULD NEED
 * SOME SCALING
 */

void Bmonomvn::DataAugment(unsigned int col, const double mu, 
			   double *beta, const double s2, 
                           const double nu)
{
  /* sanity checks */
  assert(col <= M);

  /* check to see if there is any Data Augmentation to do */
  if(!R || R->n2[col] == 0) return;

  /* get the list of rows that need infilling */
  unsigned int *R2 = R->R2[col];

  /* deal with each row */
  double ss2 = sqrt(s2);
  for(unsigned int i=0; i<R->n2[col]; i++) {
    assert((int) i < n[col]);
    double muy = mu;
    muy += linalg_ddot(col, beta, 1, Y[R2[i]], 1);
    if(theta != 0) Y[R2[i]][col] = ss2*rt(nu) + muy;
    else Y[R2[i]][col] = rnorm(muy, ss2);
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
 * Thin:
 *
 * get the regression method associated with each regression
 * model encoded as an integer 
 */

void Bmonomvn::Thin(const double thin, int *thin_out)
{
  assert(thin_out);
  for(unsigned int i=0; i<M; i++)
    thin_out[i] = blasso[i]->Thin(thin);
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

void Bmonomvn::SetSums(MVNsum *mom1, MVNsum *mom2, double *lambda2_sum, 
		       double *m_sum, MVNsum *map, MVNsum *mu_mom, 
		       MVNsum *nzS, MVNsum *nzSi)
{
  /* should also write zeros in the matrices and vectors if they are
     really meant to be sums */
  this->mom1 = mom1;
  this->mom2 = mom2;
  this->lambda2_sum = lambda2_sum;
  this->m_sum = m_sum;
  this->map = map;
  this->mu_mom = mu_mom;
  this->nzS = nzS;
  this->nzSi = nzSi;
}


/*
 * SetQP:
 *
 * set the pointers to the QPsamp strutcture with pointers to 
 * memory allocated outside the module 
 */

void Bmonomvn::SetQPsamp(QPsamp *qps)
{
  this->qps = qps;
}


/*
 * LpostMAP:
 *
 * return the log posterior probability of the maximum
 * a' posteriori (MAP) estimate of mu and S -- also
 * copies back the time index of the MAP sample
 */

double Bmonomvn::LpostMAP(int *which)
{
  *which = which_map;
  return lpost_map;
}


/*
 * get_regress:
 *
 * take a mean vector mu and covariance matrix S, with
 * components (columns) sorted according to a monotone ordering
 * and reconstruct the resulting intercepts, regression vectors
 * and variances that would have been used by the monomvn
 * algorithm -- to extract the m-th components
 *
 * IT MAY BE POSSIBLE TO SPEED THIS UP WITH SOME CLEVER 
 * ITERATIVE INVERSES
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
  int info = 0;
  info = linalg_dposv(m, s11util, s11i);
  assert(info == 0);
  if(info != 0) MYprintf(MYstdout, "linalg_dposv failed in get_regress\n");

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
     if(fabs(beta_out[i]) < sqrt(DBL_EPSILON)) beta_out[i] = 0.0; */

  /* b0 <- drop(m2 - beta %*% m1) */
  *mu_out = mu[m] - linalg_ddot(m, beta_out, 1, mu, 1);

  /* clean up */
  delete_matrix(s11util);
  delete_matrix(s11i);
}


extern "C"
{

/* for re-arranged pointers to global variables with memory in R */
void free_R_globals(void);

double **Y = NULL;
Rmiss *R = NULL;
Bmonomvn *bmonomvn = NULL;
int bmonomvn_seed_set;

/* starting structures and matrices */
MVNsum *MVNmean = NULL;
MVNsum *MVNvar = NULL;
MVNsum *MVNmap = NULL;
MVNsum *MVNmu_mom = NULL;
MVNsum *MVNnzS = NULL;
MVNsum *MVNnzSi = NULL;
QPsamp *qps = NULL;
double **S_start = NULL;


/*
 * bmonomvn_R:
 *
 * function currently used for testing the above functions
 * using R input and output
 */

void bmonomvn_R(
		/* estimation inputs */
		int *B, int *T, double *thin, int *M, int *N, double *Y_in, 
		int *n, int *R_in, double *p, int *method, int *facts_len, 
		int *facts, int *RJ, int *capm,

		/* starting values */
		int *mu_start_len, double *mu_start, 
		int *S_start_len, double *S_start_in, 
		int *ncomp_start_len, int *ncomp_start, 
		int *lambda_start_len, double *lambda_start, 

		/* other inputs */
		double *mprior, double *rd, double *theta, int *rao_s2, 
		int *economy, int *verb, int *trace, 

		/* Quadratic Programming inputs */
		int *qpnf, double *dvec, int *dmu, double *Amat, 
		double *b0, int *mu_constr, int *q, int *meq,

		/* estimation outputs: means and covars */
		double *mu_mean, double *mu_var, double *mu_cov, 
		double *S_mean, double *S_var, double* mu_map, 
		double *S_map, double *S_nz, double *Si_nz,

		/* estimation outputs: other */
		double *lpost_map, int *which_map, double *llik,
		int *llik_norm_len, double *llik_norm, int *methods, 
		int *thin_out, int *nu_len, double *nu, 
		double *lambda2_mean, double *m_mean,

		/* Quadratic Programming outputs */
		int* w_len, double *w)
{
  /* copy the vector(s) input Y and R into matrix form */
  Y = new_matrix_bones(Y_in, *N, *M);
  R = new_Rmiss_R(R_in, *N, *M);

  /* copy the vector input S_start into matrix form */
  if(*S_start_len > 0) S_start = new_matrix_bones(S_start_in, *M, *M);
  if(*mu_start_len == 0) mu_start = NULL;
  if(*ncomp_start_len == 0) ncomp_start = NULL;
  if(*lambda_start_len == 0) lambda_start = NULL;

  /* load copy the vectors mu, S_mean, mu_var, and S_var
     into the MVNsum structures */
  MVNmean = new_MVNsum_R(*M, mu_mean, S_mean);
  MVNvar = new_MVNsum_R(*M, mu_var, S_var);
  MVNmap = new_MVNsum_R(*M, mu_map, S_map);
  MVNmu_mom = new_MVNsum_R(*M, NULL, mu_cov);
  MVNnzS = new_MVNsum_R(*M, NULL, S_nz);
  MVNnzSi = new_MVNsum_R(*M, NULL, Si_nz);

  /* load Quadratic Programming inputs into the QP structure */
  if(*w_len == 0) w = NULL;
  qps = new_QPsamp_R(qpnf[0], *T, (qpnf+1), dvec, 
		     (bool) *dmu, Amat, b0, mu_constr, *q, *meq, w);

  /* deal with NULLs */
  if(*nu_len == 0) nu = NULL;
  if(*llik_norm_len == 0) llik_norm = NULL;
  if(*facts_len == 0) facts = NULL;

  /* get the random number generator state from R */
  GetRNGstate(); bmonomvn_seed_set = 1;

  /* create a new Bmonomvn object */
  bmonomvn = new Bmonomvn(*M, *N, Y, n, R, *p, *theta, *verb, (bool) (*trace));

  /* initialize the Bayesian lasso regressios with the bmonomvn module */
  bmonomvn->InitBlassos(*method, facts, *RJ, (bool) (*capm), mu_start, 
			S_start, ncomp_start, lambda_start, mprior, rd[0], 
			rd[1], (bool) (*rao_s2), (bool) *economy, 
			(bool) (*trace));

  /* do burn-in rounds */
  if(*verb) MYprintf(MYstdout, "%d burnin rounds\n", *B);
  bmonomvn->Rounds(*B, *thin, (bool) *economy, true, NULL, NULL, NULL);
  
  /* set up the mu and S sums for calculating means and variances */
  bmonomvn->SetSums(MVNmean, MVNvar, lambda2_mean, m_mean, MVNmap,
		    MVNmu_mom, MVNnzS, MVNnzSi);
  /* set the QPsamp */
  bmonomvn->SetQPsamp(qps);

  /* and now sampling rounds */
  if(*verb) MYprintf(MYstdout, "%d sampling rounds\n", *T);
  bmonomvn->Rounds(*T, *thin, (bool) *economy, false, nu, llik, llik_norm);

  /* copy back the mean and variance(s) of mu and S */
  MVN_mean(MVNmean, *T);
  MVN_var(MVNvar, MVNmean, *T);
  MVN_mean(MVNmu_mom, *T);
  MVN_mom2cov(MVNmu_mom, MVNmean);
  MVN_mean(MVNnzS, *T);
  MVN_mean(MVNnzSi, *T);

  /* get/check the MAP */
  assert(MVNmap->T == 1);
  *lpost_map = bmonomvn->LpostMAP(which_map);

  /* copy back the mean lambda2s and ms */
  scalev(lambda2_mean, *M, 1.0/(*T));
  scalev(m_mean, *M, 1.0/(*T));

  /* get the actual methods and thinning level used for each regression */
  bmonomvn->Methods(methods);
  bmonomvn->Thin(*thin, thin_out);

  /* clean up */
  delete bmonomvn;
  bmonomvn = NULL;

  /* give the random number generator state back to R */
  PutRNGstate(); bmonomvn_seed_set = 0;

  /* clean up */
  free_R_globals();
}


/*
 * free_R_globals:
 *
 * free the global variables that are necessary to re-arrange
 * the pointers to memory that are passed in from R
 */ 

void free_R_globals(void) {
  if(Y) { free(Y); Y = NULL; }
  if(R) { delete_Rmiss_R(R); R = NULL; }
  if(S_start) { free(S_start); S_start = NULL; }
  if(MVNmean) { delete_MVNsum_R(MVNmean); MVNmean = NULL; }
  if(MVNvar) { delete_MVNsum_R(MVNvar); MVNvar = NULL; }
  if(MVNmap) { delete_MVNsum_R(MVNmap); MVNmap = NULL; }
  if(MVNmu_mom) { delete_MVNsum_R(MVNmu_mom); MVNmu_mom = NULL; }
  if(MVNnzS) { delete_MVNsum_R(MVNnzS); MVNnzS = NULL; }
  if(MVNnzSi) { delete_MVNsum_R(MVNnzSi); MVNnzSi = NULL; }
  if(qps) { delete_QPsamp_R(qps); qps = NULL; }
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
      MYprintf(MYstderr, "INTERRUPT: bmonomvn model leaked, is now destroyed\n\n");
    delete bmonomvn; 
    bmonomvn = NULL; 
  }

  /* free RNG state */
  if(bmonomvn_seed_set) { PutRNGstate(); bmonomvn_seed_set = 0; }

  /* clean up matrix pointers */
  free_R_globals();
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
