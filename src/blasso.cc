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
#include "blasso.h"


/*
 * Blasso:
 *
 * the typical constructor function that initializes the
 * parameters to default values, and does all pre-processing
 * necesary to start sampling from the posterior distribution
 * of the Bayesian Lasso parameters
 */

Blasso::Blasso(const unsigned int m, const unsigned int n, double **X, 
	       double *Y, const double r, const double delta, 
	       const REG_MODEL reg_model, const bool rao_s2, 
	       const unsigned int verb)
{
  this->reg_model = reg_model;

  /* initialize the input data */ 
  InitXY(m, n, X, Y, true);

  /* initialize the regression utility */
  InitRegress();
  
  /* copy verbosity argument */
  this->verb = verb;

  /* initialize the parameters */
  this->r = r;
  this->delta = delta;

  /* this function will be modified to depend on whether OLS 
     must become lasso */
  InitParams(reg_model);

  /* copy the Rao-Blackwell option for s2 */
  this->rao_s2 = rao_s2;

  /* calculate the initial linear mean and corellation matrix */
  assert(ComputeBmuA());
}


/*
 * Blasso:
 *
 * the typical constructor function that initializes the
 * parameters to default values, and does all pre-processing
 * necesary to start sampling from the posterior distribution
 * of the Bayesian Lasso parameters
 */

Blasso::Blasso(const unsigned int m, const unsigned int n, double **X, 
	       double *Y, const double lambda2, const double s2, 
	       double *tau2i, const double r, const double delta, 
	       const double a, const double b, const bool rao_s2,
	       const bool normalize, const unsigned int verb)
{
  /* initialize the input data */ 
  InitXY(m, n, X, Y, normalize);

  /* initialize the regression utility */
  InitRegress();
  
  /* copy verbosity argument */
  this->verb = verb;

  /* initialize the parameters */
  this->r = r;
  this->delta = delta;

  /* this function will be modified to depend on whether OLS 
     must become lasso */
  InitParams(lambda2, s2, tau2i);

  /* copy the Rao-Blackwell option for s2 */
  this->rao_s2 = rao_s2;

  /* set the s2 Inv-Gamma prior */
  this->a = a;
  this->b = b;

  /* calculate the initial linear mean and corellation matrix */
  assert(ComputeBmuA());
}


/* 
 * InitXY:
 *
 * initialize the input data (X and Y)
 */

void Blasso::InitXY(const unsigned int m, const unsigned int n, 
		    double **X, double *Y, const bool normalize)
{
  /* copy the dimension parameters*/
  this->m = m;
  this->n = n;

  /* store the original X */
  this->Xorig = new_dup_matrix(X, n, m);

  /* calculate the mean of each column of X*/
  double *Xmean = new_zero_vector(m);
  wmean_of_columns(Xmean, X, n, m, NULL);
  
  /* center X */
  this->X = new_dup_matrix(X, n, m);
  center_columns(this->X, Xmean, n, m);
  free(Xmean);

  /* normalize X, like Efron & Hastie */
  /* (presumably so that [good] lambda doesn't increase with n ??) */
  this->normalize = normalize;
  if(this->normalize) {
    Xnorm = new_zero_vector(m);
    sum_of_columns_f(Xnorm, this->X, n, m, sq);
    for(unsigned int i=0; i<m; i++) Xnorm[i] = sqrt(Xnorm[i]);
    norm_columns(this->X, Xnorm, n, m);
  } else Xnorm = NULL;

  /* center Y */
  this->Y = new_dup_vector(Y, n);
  Ymean = meanv(Y, n);
  centerv(this->Y, n, Ymean); 

  /* calculate t(X) %*% X */
  XtX = new_zero_matrix(m, m);
  if(XtX) linalg_dgemm(CblasNoTrans,CblasTrans,m,m,n,1.0,
		       this->X,m,this->X,m,0.0,XtX,m);

  /* calculate t(X) %*% Y */
  XtY = new_zero_vector(m);
  if(XtY) linalg_dgemv(CblasNoTrans,m,n,1.0,this->X,m,this->Y,1,0.0,XtY,1);

  /* calculate YtY */
  YtY = linalg_ddot(n, this->Y, 1, this->Y, 1);
}


/*
 * InitRegress:
 *
 * allocate the memory necessary to do the regressions
 * required by the model
 */

void Blasso::InitRegress(void)
{
  assert(n != 0);
  Vb = new_matrix(m, m);
  bmu = new_vector(m);
  A = new_matrix(m, m);
  A_util = new_matrix(m, m);
  Ai = new_matrix(m, m);
  Xbeta_v = new_vector(n); /* only used by one ::Draw function */
  BtDi = new_vector(m);
  rn = new_vector(m);
  ABmu = new_vector(m);
  BtAB = 0.0;
}


/*
 * InitParams:
 *
 * pick (automatic) starting values for the parameters based on the type
 * of model
 */

void Blasso::InitParams(const REG_MODEL reg_model)
{
  /* set the LASSO lambda2 and tau2i initial values */
  if(reg_model == LASSO) {
    lambda2 = 1;
    tau2i = ones(m, 1.0);
  } else {
    lambda2 = 0;
    tau2i = new_zero_vector(m);
  }
  beta = new_zero_vector(m);
  s2 = 1;

  /* set the s2 Inv-Gamma prior */
  if(reg_model == LASSO) {
    a = 3.0/2.0;
    
    /* need to help with a stronger s2-prior beta parameter when p>n */
    if(m >= n) b = Igamma_inv(a, 0.99*gammafn(a), 0, 0)*YtY;
    else b = 0; 

  } else {
    a = m/2.0 + 1.0;  /* this seems necessary */
    b = 0.0;
  }
}



/*
 * InitParams:
 *
 * set specific values of the starting values, and the 
 * dynamically determine the type of model
 */

void Blasso::InitParams(const double lambda2, const double s2, double *tau2i)
{
  this->lambda2 = lambda2;
  this->s2 = s2;
  this->tau2i = new_dup_vector(tau2i, m);
  beta = new_zero_vector(m);

  if(lambda2 == 0 && sumv(tau2i, m) == 0) {
    reg_model = OLS;
  } else reg_model = LASSO;
}


/*
 * ~Blasso:
 *
 * the usual destructor function
 */

Blasso::~Blasso(void)
{
  /* clean up */
  if(beta) free(beta);
  if(bmu) free(bmu);
  if(Vb) delete_matrix(Vb);
  if(tau2i) free(tau2i);
  if(Xorig) delete_matrix(Xorig);
  if(X) delete_matrix(X);
  if(XtX) delete_matrix(XtX);
  if(normalize && Xnorm) free(Xnorm);
  if(XtY) free(XtY);
  if(Y) free(Y);
  if(A) delete_matrix(A);
  if(A_util) delete_matrix(A_util);
  if(Ai) delete_matrix(Ai);
  if(Xbeta_v) free(Xbeta_v);
  if(BtDi) free(BtDi);
  if(rn) free(rn);
}

 
/*
 * GetParams 
 * get the current values of the parameters to the pointers
 * to memory provided
 */

void Blasso::GetParams(double *beta, double *s2, double *tau2i, 
		       double *lambda2) const
{
  dupv(beta, this->beta, m);
  *s2 = this->s2;
  dupv(tau2i, this->tau2i, m);
  *lambda2 = this->lambda2;
}


/*
 * PrintParams:
 * print the current values of the parameters to the
 * specified file
 */

void Blasso::PrintParams(FILE *outfile) const
{
  myprintf(outfile, "m=%d, lambda2=%g, s2=%g\n", m, lambda2, s2);
  myprintf(outfile, "beta = ");
  printVector(beta, m, outfile, HUMAN);
  myprintf(outfile, "tau2i = ");
  printVector(tau2i, m, outfile, HUMAN);
}


/*
 * Rounds:
 *
 * perform T rounds of mcmc/Gibbs sampling the lasso parameters
 * beta, s2, tau2i and lambda -- taking thin number of them before
 * returning one set.
 */

void Blasso::Rounds(const unsigned int T, const unsigned int thin, 
		    double *lambda2, double *mu, double **beta, 
		    double *s2, double **tau2i)
{

  /* assume that the initial values reside in position 0 */
  /* do T-1 MCMC rounds */
  for(unsigned int t=0; t<T; t++) {
    
    /* do thin number of MCMC draws */
    Draw(thin);

    /* copy the sampled parameters */
    GetParams(beta[t], &(s2[t]), tau2i[t], &(lambda2[t]));
    
    /* print progress meter */
    if(verb && t > 0 && ((t+1) % 100 == 0))
      myprintf(stdout, "t=%d\n", t+1);
  }

  /* (un)-norm the beta samples, like Efron and Hastie */
  if(normalize) norm_columns(beta, Xnorm, T, m);

  /* calculate mu samples */

  /* Xbeta = X %*% t(beta), in col-major representation */
  double **Xbeta = new_zero_matrix(T,n);
  linalg_dgemm(CblasTrans,CblasNoTrans,n,T,m,1.0,Xorig,m,beta,m,0.0,Xbeta,n);

  /* mu = apply(Xbeta, 2, mean), with Xbeta in col-major representation */
  wmean_of_rows(mu, Xbeta, T, n, NULL);

  /* mu = rnorm(rep(1,Ymean), sqrt(s2/n)) - apply(Xbeta, 2, mean) */
  for(unsigned t=0; t<T; t++) mu[t] = rnorm(Ymean, sqrt(s2[t]/n)) - mu[t];

  /* clean up */
  delete_matrix(Xbeta);
}


/*
 * Draw:
 *
 * Gibbs draws for each of the bayesian lasso parameters
 * in turn: beta, s2, tau2i, and lambda2;  the thin paramteters
 * causes thin-1 (number of) draws to be burned first
 */

void Blasso::Draw(const unsigned int thin)
{
  for(unsigned int t=0; t<thin; t++) {

    /* depends on pre-calculated bmu, Vb, etc, which depends
       on tau2i */
    DrawBeta();

    /* choose the type of s2 GS update */
    if(rao_s2) DrawS2Margin();  /* depends on bmu and Vb but not beta */
    else DrawS2();  /* depends on beta */

    /* draw latent variables, and update Bmu and Vb, etc. */
    DrawTau2i();

    /* only depends on tau2i */
    DrawLambda2();
  }
}


/*
 * Draw:
 *
 * Gibbs draws for each of the bayesian lasso parameters
 * in turn: beta, s2, tau2i, and lambda2; the thin paramteters
 * causes thin-1 (number of) draws to be burned first, and
 * copy them out to the pointers/memory passed in
 */

void Blasso::Draw(const unsigned int thin, double *lambda2, double *mu, 
		  double *beta,  double *s2, double *tau2i)
{
  // PrintInputs(stdout);

  /* do thin number of MCMC draws */
  Draw(thin);

  /* copy the sampled parameters */
  GetParams(beta, s2, tau2i, lambda2);

  /* (un)-norm the beta samples, like Efron and Hastie */
  if(normalize && m > 0) normv(beta, m, Xnorm);

  if(m > 0) {
    /* Xbeta = X %*% beta, in col-major representation */
    linalg_dgemv(CblasTrans,m,n,1.0,Xorig,m,beta,1,0.0,Xbeta_v,1);
    
    /* mu = mean(Xbeta) */
    *mu = meanv(Xbeta_v, n);
  } else *mu = 0;
  
  /* mu = rnorm(Ymean, sqrt(s2/n)) - mean(Xbeta) */
  *mu = rnorm(Ymean, sqrt((*s2)/n)) - (*mu);
}


/*
 * ComputeBmuA:
 *
 * compute the (mle) linear parameters Bmu (the mean)
 * A (the correllation matrix) and its inverse and
 * calculate the product t(Bmu) %*% A %*% Bmu
 */

bool Blasso::ComputeBmuA(void)
{
  if(m == 0) return true;

  /* compute: A = XtX + Dtaui */
  dup_matrix(A, XtX, m, m);
  for(unsigned int i=0; i<m; i++) A[i][i] += tau2i[i];

  /* Ai = inv(A) */
  dup_matrix(A_util, A, m, m);
  id(Ai,m);
  int info = linalg_dposv(m, A_util, Ai);
  //assert(info == 0);
  /* now A_util is useless */
  
  /* unsuccessful inverse */
  if(info != 0) return false;
  
  /* compute: Bmu = Ai %*% Xt %*% ytilde */
  linalg_dsymv(m, 1.0, Ai, m, XtY, 1, 0.0, bmu, 1);

  /* t(Bmu) %*% (A) %*% Bmu */
  linalg_dsymv(m, 1.0, A, m, bmu, 1, 0.0, ABmu, 1);
  BtAB = linalg_ddot(m, bmu, 1, ABmu, 1);

  return true;
}


/*
 * DrawBeta:
 *
 * Gibbs draw for the beta m-vector conditional on the
 * other Bayesian lasso parameters -- assumes that updated
 * bmu and Ai have been precomputed by ComputeBmuA()
 */

void Blasso::DrawBeta(void)
{
  if(m == 0) return;

  /* compute: Vb = s2*Ai */
  dup_matrix(Vb, Ai, m, m);
  scalev(*Vb, m*m, s2);
  
  /* draw from the multivariate normal distribution */
  linalg_dpotrf(m, Vb);
  for(unsigned int i=0; i<m; i++) rn[i] = norm_rand();
  mvnrnd(beta, bmu, Vb, rn, m);
}


/*
 * DrawS2Margin:
 *
 * Gibbs draw for the s2 scalar conditional on the
 * other Bayesian lasso parameters -- assumes that updated
 * bmu and Ai have been precomputed by ComputeBmuA() -- does
 * not depend on beta because it has been integrated out
 */

void Blasso::DrawS2Margin(void)
{
  /* shape = (n-1)/2 + m/2 */
  double shape = a;
  if(reg_model == LASSO) shape += (n-1)/2.0;//+ m/2.0;
  else shape += n/2.0 - m/2.0;
  
  /* rate = (X*beta - Y)' (X*beta - Y) / 2 + B'DB / 2*/
  double scale = b + (YtY - BtAB)/2.0;
  
  /* draw the sample and return it */
  s2 = 1.0/rgamma(shape, 1.0/scale);

  /* check for a problem */
  if(scale <= 0) {
    PrintParams(stdout);
    myprintf(stdout, "YtY=%.20f, BtAB=%.20f\n", YtY, BtAB);
    assert(scale > 0);
  }
}


/*
 * DrawS2:
 *
 * Gibbs draw for the s2 scalar conditional on the
 * other Bayesian lasso parameters, depends on beta
 */

void Blasso::DrawS2(void)
{
  /* resid = X*beta - Y */
  double *resid = new_dup_vector(Y, n);
  if(m > 0) linalg_dgemv(CblasTrans,m,n,-1.0,X,m,beta,1,1.0,resid,1);
  
  /* sums2 = (X*beta - Y)' (X*beta - Y); then resid not needed */
  double sums2 = sum_fv(resid, n, sq);

  /* BtDB = beta'D beta/tau2 as long as lambda != 0 */
  /* MIGHT EVENTRUALLY NEED TO ALLOW ZERO-LAMBDA WITH FIXED TAU2I */
  double BtDiB;
  if(m > 0 && reg_model == LASSO) {
    dupv(BtDi,beta, m);
    scalev2(BtDi, m, tau2i);
    BtDiB = linalg_ddot(m, BtDi, 1, beta, 1);
  } else BtDiB = 0.0;
    
  /* shape = (n-1)/2 + m/2 */
  double shape = a;
  if(reg_model == LASSO) shape += (n-1)/2.0 + m/2.0;
  else shape += (n-1)/2.0; // - m/2.0;
  
  /* rate = (X*beta - Y)' (X*beta - Y) / 2 + B'DB / 2*/
  double scale = b + sums2/2.0 + BtDiB/2.0;
  
  /* draw the sample and return it */
  s2 = 1.0/rgamma(shape, 1.0/scale);

  /* check for a problem */
  if(scale <= 0) {
    PrintParams(stdout);
    myprintf(stdout, "sums2=%g, BtDiB=%g\n", sums2, BtDiB);
    assert(scale > 0);
  }

  free(resid);
}


/*
 * DrawTau2i:
 *
 * Gibbs draw for the inverse tau2 m-vector (latent variables)
 * conditional on the other Bayesian lasso parameters
 */

void Blasso::DrawTau2i(void)
{
  double l_numer, l_mup;

  /* special case where we're not actually doing lasso */
  if(m == 0 || lambda2 <= 0) return;

  /* part of the mu parameter to the inv-gauss distribution */
  l_numer = log(lambda2) + log(s2);

  for(unsigned int j=0; j<m; j++) {
      
    /* the rest of the mu parameter */
    l_mup = 0.5*l_numer - log(fabs(beta[j])); //sqrt(numer/sq(beta[j]));
    
    /* sample from the inv-gauss distn */
    tau2i[j] = rinvgauss(exp(l_mup), lambda2);    
    
    /* check to make sure there were no numerical problems */
    if(tau2i[j] <= 0) {
#ifdef DEBUG
      myprintf(stdout, "j=%d, m=%d, n=%d, l2=%g, s2=%g, beta=%g, tau2i=%g\n", 
	       j, m, n, lambda2, s2, beta[j], tau2i[j]);
#endif
      tau2i[j] = 0;
    }
  }
  
  /* Since tau2i has changed, we need to recompute the linear
     parameters */
  assert(ComputeBmuA() && (YtY - BtAB > 0));
}


/*
 * DrawLambda2:
 *
 * Gibbs draw for the lambda2 scalar conditional on the
 * other Bayesian lasso parameters
 */

void Blasso::DrawLambda2(void)
{
  if(m == 0 || lambda2 <= 0) return;

  double shape = (double) m + r;
  double rate = 0.0;
  for(unsigned int i=0; i<m; i++) {
    if(tau2i[i] == 0) {shape--; continue;}  /* for numerical problems */
    rate += 1.0/tau2i[i];
  }
  rate = rate/2.0 + delta;
  
  lambda2 = rgamma(shape, 1.0/rate);
}


/*
 * PrintInputs:
 *
 * print the design matrix (X) and responses (Y)
 */

void Blasso::PrintInputs(FILE *outfile) const
{
  /* print the design matrix */
  if(X) {
    myprintf(outfile, "X =\n");
    printMatrix(X, n, m, outfile);
  } else {
    myprintf(outfile, "X = NULL\n");
  }

  /* print the response vector */
  myprintf(outfile, "Y = ");
  printVector(Y, n, outfile, HUMAN);
}


/*
 * RegType:
 *
 * return the regression type
 */

REG_MODEL Blasso::RegModel(void)
{
  return reg_model;
}


/*
 * mvnrnd:
 * 
 * draw from a umltivariate normal mu is an n-array, 
 * and cov is an n*n array whose lower triabgular 
 * elements are a cholesky decomposition and the 
 * diagonal has the pivots. requires a choleski 
 * decomposition be performed first.
 * code from Herbie
 */

void mvnrnd(double *x, double *mu, double **cov, double *rn, 
	    const unsigned int n)
{
  unsigned int i,j;
  for(j=0;j<n;j++) {
    x[j] = mu[j];
    for(i=0;i<j+1;i++) x[j] += cov[j][i] * rn[i];
  }
}


/*
 * rinvgauss:
 *
 * Michael/Schucany/Haas Method for generating Inverse Gaussian
 * random variable, as given in Gentle's book on page 193
 */

double rinvgauss(const double mu, const double lambda)
{
  double u, y, x1, mu2, l2;

  y = sq(norm_rand());
  mu2 = sq(mu);
  l2 = 2*lambda;
  x1 = mu + mu2*y/l2 - (mu/l2)* sqrt(4*mu*lambda*y + mu2*sq(y));

  u = unif_rand();
  if(u <= mu/(mu + x1)) return x1;
  else return mu2/x1;
}


/*
 * Igamma_inv:
 *
 * incomplete gamma inverse function from UCS
 */

double Igamma_inv(const double a, const double y, const int lower, 
		  const int ulog) 
{
  double r;
  if(ulog) r = Rgamma_inv(a, y - Cgamma(a, ulog), lower, ulog);
  else r = Rgamma_inv(a, y / Cgamma(a, ulog), lower, ulog);
  assert(!isnan(r));
  /* myprintf(stdout, "Rgamma_inv: a=%g, y=%g, lower=%d, ulog=%d, r=%g\n", 
     a, y, lower, ulog, r); */
  return(r);
}

/* 
 * Cgamma: 
 *
 * (complete) gamma function and its logarithm (all logarithms are base 10) 
 * from UCS
 */

double Cgamma(const double a, const int ulog) 
{
  double r;
  if(ulog) r = lgammafn(a) / M_LN10;
  else r = gammafn(a);
  /* myprintf(stdout, "Cgamma: a=%g, ulog=%d, r=%g\n", a, ulog, r); */
  assert(!isnan(r));
  return(r);
}


/*
 * Rgamma_inv:
 *
 * regularized gamma inverse function from UCS
 */

double Rgamma_inv(const double a, const double y, const int lower, 
		  const int ulog) 
{
  double r;
  if(ulog) r = qgamma(y*M_LN10, a, /*scale=*/ 1.0, lower, ulog);
  else r = qgamma(y, a, /*scale=*/ 1.0, lower, ulog);
  /*myprintf(stdout, "Rgamma_inv: a=%g, y=%g, lower=%d, ulog=%d, r=%g\n", 
    a, y, lower, ulog, r); */
  assert(!isnan(r)); 
  return(r);
}


extern "C"
{
/*
 * lasso_draw_R
 *
 * function currently used for testing the above functions
 * using R input and output
 */

void blasso_R(int *T, int *thin, int *m, int *n, double *X_in, 
	      double *Y, double *lambda2, double *mu, double *beta, 
	      double *s2, double *tau2i, double *r, double *delta, 
	      double *a, double *b, int *rao_s2, int *normalize, int *verb)
{
  double **X, **beta_mat, **tau2i_mat;
  int i;

  /* copy the vector input X into matrix form */
  X = (double **)  malloc(sizeof(double*) * (*n));
  X[0] = X_in;
  for(i=1; i<(*n); i++) X[i] = X[i-1] + (*m);

  /* get the random number generator state from R */
  GetRNGstate();

  /* initialize a matrix for beta samples */
  beta_mat = (double **) malloc(sizeof(double*) * (*T));
  beta_mat[0] = beta;
  for(i=1; i<(*T); i++) beta_mat[i] = beta_mat[i-1] + (*m);

  /* initialize a matrix for tau2i samples */
  tau2i_mat = (double **)  malloc(sizeof(double*) * (*T));
  tau2i_mat[0] = tau2i;
  for(i=1; i<(*T); i++) tau2i_mat[i] = tau2i_mat[i-1] + (*m);

  Blasso *blasso = new Blasso(*m, *n, X, Y, lambda2[0], s2[0], tau2i_mat[0],
			      *r, *delta, *a, *b, (bool) *rao_s2, 
			      (bool) *normalize, *verb);

  /* Gibbs draws for the parameters */
  blasso->Rounds((*T)-1, *thin, &(lambda2[1]), &(mu[1]), &(beta_mat[1]), 
		 &(s2[1]), &(tau2i_mat[1]));

  delete blasso;

  /* give the random number generator state back to R */
  PutRNGstate();

  /* clean up */
  free(X);
  free(beta_mat);
  free(tau2i_mat);
}


/*
 * Igamma_inv_R:
 *
 * function to test the Igamma_inv function in R to compare
 * with the Igamma.inv function in the UCS library
 */
  
void Igamma_inv_R(double *a, double *y, int *lower, int *ulog, double *result) 
{
  *result = Igamma_inv(*a, *y, *lower, *ulog);
}
}
