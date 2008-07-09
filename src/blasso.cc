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

#define DEBUG

/*
 * Blasso:
 *
 * the typical constructor function that initializes the
 * parameters to default values, and does all pre-processing
 * necesary to start sampling from the posterior distribution
 * of the Bayesian Lasso parameters
 */

Blasso::Blasso(const unsigned int M, const unsigned int n, double **X, 
	       double *Y, const bool RJ, const unsigned int Mmax, 
	       const double r, const double delta, 
	       const REG_MODEL reg_model, const bool rao_s2, 
	       const unsigned int verb)
{
  /* sanity checks */
  if(Mmax >= n) assert(RJ || reg_model == LASSO);

  /* copy RJ setting */
  this->RJ = RJ;

  /* initialize the active set of columns of X */
  InitIndicators(M, Mmax, NULL, NULL);

  /* copy the Rao-Blackwell option for s2 */
  this->rao_s2 = rao_s2;

  /* initialize the input data */ 
  InitXY(n, X, Y, true);

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

  /* initialize the residula vector */
  resid = new_dup_vector(Y, n);

  /* calculate the initial linear mean and corellation matrix */
  assert(Compute());

  /* only used by one ::Draw function */
  Xbeta_v = new_vector(n);

  /* initialize beta so that it is non-zero */
  for(unsigned int i=0; i<m; i++) breg->beta[i] = 1.0;
}


/*
 * Blasso:
 *
 * the typical constructor function that initializes the
 * parameters to default values, and does all pre-processing
 * necesary to start sampling from the posterior distribution
 * of the Bayesian Lasso parameters
 */

Blasso::Blasso(const unsigned int M, const unsigned int n, double **X, 
	       double *Y, const bool RJ, const unsigned int Mmax, 
	       double *beta, const double lambda2, 
	       const double s2, double *tau2i, const double r, 
	       const double delta, const double a, const double b, 
	       const bool rao_s2, const bool normalize, 
	       const unsigned int verb)
{
  /* copy RJ setting */
  this->RJ = RJ;

  /* initialize the active set of columns of X */
  InitIndicators(M, Mmax, beta, tau2i);

  /* copy the Rao-Blackwell option for s2 */
  this->rao_s2 = rao_s2;

  /* initialize the input data */ 
  InitXY(n, X, Y, normalize);

  /* initialize the regression utility */
  InitRegress();
  
  /* copy verbosity argument */
  this->verb = verb;

  /* initialize the parameters */
  this->r = r;
  this->delta = delta;

  /* this function will be modified to depend on whether OLS 
     must become lasso */
  InitParams(beta, lambda2, s2, tau2i); 

  /* initialize the residula vector */
  resid = new_dup_vector(Y, n);

  /* set the s2 Inv-Gamma prior */
  this->a = a;
  this->b = b;

  /* calculate the initial linear mean and corellation matrix */
  assert(Compute());

  Xbeta_v = NULL;
}


/*
 * InitIndicators:
 *
 * set the total number of colums, M, and then set the
 * initial number of non-zero columns, m
 */

void Blasso::InitIndicators(const unsigned int M, unsigned int Mmax, 
			    double *beta, double *tau2i)
{
  /* copy the dimension parameters*/
  this->M = M;
  this->Mmax = Mmax;

  /* sanity checks */
  assert(Mmax <= M);
  if(!RJ) assert(Mmax == M);

  /* find out which betas are non-zero, thus setting m */
  pb = (bool*) malloc(sizeof(bool) * M);

  if(beta != NULL) {
    assert(tau2i != NULL);
    m = 0;
    for(unsigned int i=0; i<M; i++) {
      if(beta[i] != 0) { 
	pb[i] = true; m++; 
	// assert(tau2i[i] > 0);
      }
      else pb[i] = false;
      assert(m <= Mmax);
    }
  } else {
    for(unsigned int i=0; i<Mmax; i++) pb[i] = true;
    m = Mmax;
    for(unsigned int i=Mmax; i<M; i++) pb[i] = false;
  }

  /* allocate the column indicators and fill them */
  this->pin = new_ivector(m);
  this->pout = new_ivector(M-m);
  unsigned int j = 0, k = 0;
  for(unsigned int i=0; i<M; i++) {
    if(pb[i]) pin[j++] = i;
    else pout[k++] = i;
  }
}


/* 
 * InitXY:
 *
 * initialize the input data (X and Y)
 */

void Blasso::InitXY(const unsigned int n, double **X, double *Y, 
		    const bool normalize)
{
  this->n = n;

  /* copy the input matrix */
  this->Xorig = new_dup_matrix(X, n, M);

  /* calculate the mean of each column of X*/
  double *Xmean = new_zero_vector(M);
  wmean_of_columns(Xmean, this->Xorig, n, M, NULL);
  
  /* center X */
  this->X = new_dup_matrix(X, n, M);
  center_columns(this->X, Xmean, n, M);
  free(Xmean);

  /* normalize X, like Efron & Hastie */
  /* (presumably so that [good] lambda doesn't increase with n ??) */
  this->normalize = normalize;
  if(this->normalize) {
    Xnorm = new_zero_vector(M);
    sum_of_columns_f(Xnorm, this->X, n, M, sq);
    for(unsigned int i=0; i<M; i++) Xnorm[i] = sqrt(Xnorm[i]);
    norm_columns(this->X, Xnorm, n, M);
  } else Xnorm = NULL;

  /* center Y */
  this->Y = new_dup_vector(Y, n);
  Ymean = meanv(Y, n);
  centerv(this->Y, n, Ymean); 

  /* extract the active columns of X */
  Xp = new_p_submatrix(pin, this->X, n, m);

  /* calculate t(X) %*% X */
  XtX = new_zero_matrix(m, m);
  if(XtX) linalg_dgemm(CblasNoTrans,CblasTrans,m,m,n,1.0,
		       Xp,m,Xp,m,0.0,XtX,m);

  /* calculate t(X) %*% Y */
  XtY = new_zero_vector(m);
  if(XtY) linalg_dgemv(CblasNoTrans,m,n,1.0,Xp,m,this->Y,1,0.0,XtY,1);

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

  /* allocate a new regression utility structure */
  breg = new_BayesReg(m, XtX);

  /* allocate the other miscellaneous vectors needed for
     doing the regressions */
  if(!rao_s2) BtDi = new_vector(m);
  else BtDi = NULL;
  rn = new_vector(M);
}


/*
 * new_BayesReg:
 *
 * allocate a new regression utility structure
 */

BayesReg* new_BayesReg(unsigned int m, double **XtX)
{
  BayesReg *breg = (BayesReg*) malloc(sizeof(struct bayesreg));
  breg->m = m;
  breg->Vb = new_matrix(m, m);
  breg->Vb_state = NOINIT;
  breg->bmu = new_vector(m);
  breg->A = new_dup_matrix(XtX, m, m);
  breg->A_util = new_matrix(m, m);
  breg->Ai = new_matrix(m, m);
  breg->ABmu = new_vector(m);
  breg->BtAB = 0.0;
  breg->beta = new_zero_vector(m);
  breg->lprob = 0.0;
  return(breg);
}


/*
 * InitParams:
 *
 * pick (automatic) starting values for the parameters based on the type
 * of model
 */

void Blasso::InitParams(const REG_MODEL reg_model)
{
  this->reg_model = reg_model;

  /* set the LASSO lambda2 and tau2i initial values */
  if(reg_model == LASSO) {
    lambda2 = 1;
    tau2i = ones(m, 1.0);
  } else {
    lambda2 = 0;
    tau2i = new_zero_vector(m);
  }

  /* initialize regression coefficients */
  s2 = 1;

  /* default setting when not big-p-small n */
  a = b = 0;

  /* set the s2 Inv-Gamma prior */
  if(reg_model == LASSO) {
    
    /* need to help with a stronger s2-prior beta parameter when p>n */
    if((!RJ || (RJ && lambda2 <= 0)) && M >= n) {
      a = 3.0/2.0;
      b = Igamma_inv(a, 0.99*gammafn(a), 0, 0)*YtY;
    }
  }
}



/*
 * InitParams:
 *
 * set specific values of the starting values, and the 
 * dynamically determine the type of model
 */

void Blasso::InitParams(double *beta, const double lambda2, 
			const double s2, double *tau2i)
{
  this->lambda2 = lambda2;
  this->s2 = s2;

  this->tau2i = new_sub_vector(pin, tau2i, m);
  breg->beta = new_sub_vector(pin, beta, m);
 
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
    if(tau2i) free(tau2i);
  if(Xorig) delete_matrix(Xorig);
  if(X) delete_matrix(X);
  if(Xp) delete_matrix(Xp);
  if(XtX) delete_matrix(XtX);
  if(normalize && Xnorm) free(Xnorm);
  if(XtY) free(XtY);
  if(Y) free(Y);
  if(resid) free(resid);

  /* free the regression utility */
  if(breg) delete_BayesReg(breg);

  /* free extra regression utility vectors */
  if(Xbeta_v) free(Xbeta_v);
  if(BtDi) free(BtDi);
  if(rn) free(rn);

  /* free the boolean column indicators */
  if(pb) free(pb);
  if(pin) free(pin);
  if(pout) free(pout);

}


/*
 * delete_BayesReg:
 *
 * free the space used by the regression utility
 * structure
 */

void delete_BayesReg(BayesReg* breg)
{
  if(breg->A) delete_matrix(breg->A);
  if(breg->A_util) delete_matrix(breg->A_util);
  if(breg->Ai) delete_matrix(breg->Ai);
  if(breg->ABmu) free(breg->ABmu);
  if(breg->bmu) free(breg->bmu);
  if(breg->Vb) delete_matrix(breg->Vb);
  if(breg->beta) free(breg->beta);
  free(breg);
}

 
/*
 * GetParams 
 * get the current values of the parameters to the pointers
 * to memory provided -- assumes beta is an m-vector
 */

void Blasso::GetParams(double *beta, int *m, double *s2, double *tau2i, 
		       double *lambda2) const
{
  *m = this->m;
  zerov(beta, M);
  if(this->m > 0) copy_p_vector(beta, pin, breg->beta, this->m);
  // dupv(beta, this->beta, m);
  *s2 = this->s2;
  for(unsigned int i=0; i<M; i++) tau2i[i] = -1.0;
  if(this->m > 0) copy_p_vector(tau2i, pin, this->tau2i, this->m);
  // dupv(tau2i, this->tau2i, m);
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
  printVector(breg->beta, m, outfile, HUMAN);
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
		    double *lambda2, double *mu, double **beta, int *m,
		    double *s2, double **tau2i, double *lpost)
{

  /* assume that the initial values reside in position 0 */
  /* do T-1 MCMC rounds */
  for(unsigned int t=0; t<T; t++) {
    
    /* do thin number of MCMC draws */
    Draw(thin);

    /* copy the sampled parameters */
    GetParams(beta[t], &(m[t]), &(s2[t]), tau2i[t], &(lambda2[t]));
    
    /* get the log posterior */
    lpost[t] = this->lpost;

    /* print progress meter */
    if(verb && t > 0 && ((t+1) % 100 == 0))
      myprintf(stdout, "t=%d, m=%d\n", t+1, this->m);
  }

  /* (un)-norm the beta samples, like Efron and Hastie */
  if(normalize) norm_columns(beta, Xnorm, T, M);

  /* calculate mu samples */

  /* Xbeta = X %*% t(beta), in col-major representation */
  double **Xbeta = new_zero_matrix(T,n);
  linalg_dgemm(CblasTrans,CblasNoTrans,n,T,M,1.0,Xorig,M,beta,M,0.0,Xbeta,n);

  /* mu = apply(Xbeta, 2, mean), with Xbeta in col-major representation */
  wmean_of_rows(mu, Xbeta, T, n, NULL);

  /* mu = rnorm(rep(1,Ymean), sqrt(s2/n)) - apply(Xbeta, 2, mean) */
  for(unsigned t=0; t<T; t++) {
    double sd = sqrt(s2[t]/n);
    double mu_adj = rnorm(Ymean, sd);
    mu[t] = mu_adj - mu[t];
    lpost[t] += dnorm(mu_adj, Ymean, sd, 1);
  }

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
  /* sanity check */
  assert(thin > 0);
  for(unsigned int t=0; t<thin; t++) {

    /* draw latent variables, and update Bmu and Vb, etc. */
    DrawTau2i();
    
    /* depends on pre-calculated bmu, Vb, etc, which depends
       on tau2i -- breg->Vb then becomes decomposed */
    DrawBeta();

    /* resid = X*beta - Y */
    dupv(resid, Y, n);
    if(m > 0) linalg_dgemv(CblasTrans,m,n,-1.0,Xp,m,breg->beta,1,1.0,resid,1);

    /* choose the type of s2 GS update */
    if(rao_s2) {
      DrawS2Margin();  /* depends on bmu and Vb but not beta */
    } else DrawS2();  /* depends on beta */

    /* only depends on tau2i */
    DrawLambda2();

    /* propose to add or remove a column from the model */
    if(RJ) {
      /* first tally the log posterior value of this sample */
      lpost = logPosterior();
      RJmove();
    }
  }

  /* calculate the log posterior if it hasn't already been done */
  if(!RJ) lpost = logPosterior();
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
		  double *beta, int *m, double *s2, double *tau2i, 
		  double *lpost)
{
  // PrintInputs(stdout);

  /* do thin number of MCMC draws */
  Draw(thin);

  /* copy the sampled parameters */
  GetParams(beta, m, s2, tau2i, lambda2);

  /* (un)-norm the beta samples, like Efron and Hastie */
  if(normalize && this->m > 0) normv(beta, M, Xnorm);

  if(this->m > 0) {
    /* Xbeta = X %*% beta, in col-major representation */
    linalg_dgemv(CblasTrans,M,n,1.0,Xorig,M,beta,1,0.0,Xbeta_v,1);
    
    /* mu = mean(Xbeta) */
    *mu = meanv(Xbeta_v, n);
  } else *mu = 0;
  
  /* mu = rnorm(Ymean, sqrt(s2/n)) - mean(Xbeta) */
  double sd = sqrt((*s2)/n);
  double mu_adj = rnorm(Ymean, sd);
  *mu = mu_adj - (*mu);

  /* calculate the log posterior */
  *lpost = this->lpost;
  *lpost += dnorm(mu_adj, Ymean, sd, 1);
}


/*
 * RJmove:
 *
 * propose to add or remove a column from the design 
 * matrix X, thus adding or removing a coefficient beta (and
 * corresponding tau2i) from the model
 */

void Blasso::RJmove(void)
{
  /* no proposals here if X is empty */
  if(M == 0) return;
  
  /* choose whether to try to increase or decrease the
     size of the model */
  if(m == Mmax) RJdown(0.5);
  else if(m == 0) RJup(0.5);
  else if(unif_rand() < 0.5) RJup(1.0);
  else RJdown(1.0);
}


/*
 * RJup:
 *
 * try to add another column to the model 
 */

void Blasso::RJup(double q)
{
  /* sanity checks */
  assert(m != M);

  /* randomly select a column for addition */
  int iout = (int) ((M-m) * unif_rand());
  int col = pout[iout];
  double *xnew = new_vector(n);
  for(unsigned int i=0; i<n; i++) xnew[i] = X[i][col];
  q *= ((double)(M-m))/(m+1);

  /* randomly propose a new tau2i-component */
  tau2i = (double*) realloc(tau2i, sizeof(double)*(m+1));
  double t2;
  if(lambda2 == 0) t2 = 1.0;
  else t2 = rexp(2.0/lambda2);
  tau2i[m] = 1.0 / t2;

  /* add the new row and column to A and XtX */
  double **XtX_new = new_matrix(m+1, m+1);
  dup_matrix(XtX_new, XtX, m, m);

  /* XtX[m][m] = t(x) %*% x */
  XtX_new[m][m] = linalg_ddot(n, xnew, 1, xnew, 1);

  /* get the new row of XtX_new[col,] = X[,col] %*% X[,1:m] */
  for(unsigned int j=0; j<m; j++) {
    XtX_new[m][j] = 0;
    for(unsigned int i=0; i<n; i++) XtX_new[m][j] += Xp[i][j]*xnew[i];
  }

  /* copy the rows to the columns */
  dup_col(XtX_new, m, XtX_new[m], m);

  /* add a new component to XtY */
  XtY = (double*) realloc(XtY, sizeof(double)*(m+1));
  XtY[m] = linalg_ddot(n, xnew, 1, Y, 1);

  /* allocate new regression stuff */
  /* diagonal of A is taken care of inside of compute_BayesReg() */
  BayesReg *breg_new = new_BayesReg(m+1, XtX_new);

  /* compute the new regression quantities */
  bool success = compute_BayesReg(m+1, XtX_new, XtY, tau2i, s2, breg_new);
  assert(success);

  /* draw the new beta vector */
  draw_beta(m+1, breg_new, s2, rn, true);

  /* calculate new residual vector */
  double *resid_new = new_dup_vector(Y, n);
  if(m > 0) linalg_dgemv(CblasTrans,m,n,-1.0,Xp,m,breg_new->beta,1,1.0,resid_new,1);
  linalg_daxpy(n, 0.0 - breg_new->beta[m], xnew, 1, resid_new, 1);

  /* calculate the new log_posterior */
  double lpost_new = 
    log_posterior(n, m+1, resid_new, breg_new->beta, s2, tau2i, 
		  lambda2, a, b, r, delta);

  /* calculate the posterior ratio */
  double lalpha = lpost_new - lpost;

  /* add in the forwards probabilities */
  lalpha -= breg_new->lprob;
  if(lambda2 != 0) lalpha -= dexp(t2,2.0/lambda2,1);

  /* add in the backwards probabilities */
  lalpha += breg->lprob;

  /* MH accept or reject */
  if(unif_rand() < exp(lalpha)*q) { /* accept */

    /* copy the new regression utility */
    delete_BayesReg(breg); breg = breg_new;
    free(resid); resid = resid_new;
    delete_matrix(XtX); XtX = XtX_new;

    /* other copies */
    if(BtDi) BtDi = (double*) realloc(BtDi, sizeof(double) * (m+1));
    lpost = lpost_new;

    /* add another column to the effective design matrix */
    Xp = new_bigger_matrix(Xp, n, m, n, m+1);
    dup_col(Xp, m, xnew, n);

    add_col(iout, col);
    /* myprintf(stdout, "accepted RJ-up col=%d, pratio=%g, alpha=%g\n", 
       col, exp(lpost_new - lpost), exp(lalpha)); */

  } else { /* reject */
    
    /* realloc vectors */
    tau2i = (double*) realloc(tau2i, sizeof(double)*m);
    XtY = (double*) realloc(XtY, sizeof(double)*m);
    
    /* free new regression utility */
    delete_BayesReg(breg_new);
    delete_matrix(XtX_new);
    free(resid_new);

    /* myprintf(stdout, "rejected RJ-up col=%d, pratio=%g, alpha=%g\n", 
       col, exp(lpost_new - lpost), exp(lalpha));  */
  }

  /* clean up */
  free(xnew);
}



/*
 * RJdown:
 *
 * try to remove a column to the model 
 */

void Blasso::RJdown(double q)
{
 /* sanity checks */
  assert(m != 0);

  /* select a column for deletion */
  int iin = (int) (m * unif_rand());
  int col = pin[iin];
  q *= ((double)m)/(M-m+1);

  /* make the new design matrix with one fewer column */
  double **Xp_new = new_dup_matrix(Xp, n, m-1);
  if(iin != ((int)m)-1)
    for(unsigned int i=0; i<n; i++) Xp_new[i][iin] = Xp[i][m-1];
  
  /* select corresponding tau2i-component */
  double t2 = 1.0/tau2i[iin];
  tau2i[iin] = tau2i[m-1];
  tau2i = (double*) realloc(tau2i, sizeof(double)*(m-1));
  
  /* add the new row and column to A and XtX */
  double ** XtX_new = new_zero_matrix(m-1, m-1);
  if(XtX_new) linalg_dgemm(CblasNoTrans,CblasTrans,m-1,m-1,n,1.0,
			   Xp_new,m-1,Xp_new,m-1,0.0,XtX_new,m-1);

  /* remove component of XtY */
  double xty = XtY[iin];
  XtY[iin] = XtY[m-1];
  XtY = (double*) realloc(XtY, sizeof(double)*(m-1));

  /* allocate new regression stuff */
  BayesReg *breg_new = new_BayesReg(m-1, XtX_new);

  /* compute the new regression quantities */
  bool success = compute_BayesReg(m-1, XtX_new, XtY, tau2i, s2, breg_new);
  assert(success);

  /* draw the new beta vector */
  draw_beta(m-1, breg_new, s2, rn, true);

  /* calculate new residual vector */
  double *resid_new = new_dup_vector(Y, n);
  if(m-1 > 0)
    linalg_dgemv(CblasTrans,m-1,n,-1.0,Xp_new,m-1,breg_new->beta,1,1.0,resid_new,1);

  /* calculate the new log_posterior */
  double lpost_new = 
    log_posterior(n, m-1, resid_new, breg_new->beta, s2, tau2i, 
		  lambda2, a, b, r, delta);

  /* calculate the posterior ratio */
  double lalpha = lpost_new - lpost;

  /* add in the forwards probabilities */
  lalpha -= breg_new->lprob;

  /* add in the backwards probabilities */
  lalpha += breg->lprob;
  if(lambda2 != 0) lalpha += dexp(t2,2.0/lambda2,1);

  /* MH accept or reject */
  if(unif_rand() < exp(lalpha)*q) { /* accept */

    /* myprintf(stdout, "accepted RJ-down, col=%d, lnew=%g, lold=%g, pr=%g, alpha=%g\n", 
       col, lpost_new, lpost, exp(lpost_new - lpost), exp(lalpha)); */

    /* copy the new regression utility */
    delete_BayesReg(breg); breg = breg_new;
    free(resid); resid = resid_new;
    delete_matrix(XtX); XtX = XtX_new;

    /* other */
    if(BtDi) BtDi = (double*) realloc(BtDi, sizeof(double) * (m-1));
    lpost = lpost_new;

    /* permanently remove the column to the effective design matrix */
    delete_matrix(Xp); Xp = Xp_new;
    remove_col(iin, col);

  } else { /* reject */

    /* realloc vectors */
    tau2i = (double*) realloc(tau2i, sizeof(double)*m);
    tau2i[m-1] = tau2i[iin]; tau2i[iin] = 1.0/t2;
    XtY = (double*) realloc(XtY, sizeof(double)*m);
    XtY[m-1] = XtY[iin]; XtY[iin] = xty;
    
    /* free new regression utility */
    delete_BayesReg(breg_new);
    free(resid_new);
    delete_matrix(XtX_new); delete_matrix(Xp_new);

    /* myprintf(stdout, "rejected RJ-down, col=%d, lnew=%g, lold=%g, prat=%g, alpha=%g\n", 
       col, lpost_new, lpost, exp(lpost_new - lpost), exp(lalpha)); */
  }
}


/*
 * Compute:
 *
 * compute the (mle) linear parameters Bmu (the mean)
 * A (the correllation matrix) and its inverse and
 * calculate the product t(Bmu) %*% A %*% Bmu -- see
 * the corresponding C function below
 */

bool Blasso::Compute(void)
{
  if(m == 0) return true;

  bool ret = compute_BayesReg(m, XtX, XtY, tau2i, s2, breg); 
  
  return ret && (YtY - breg->BtAB > 0);
}


/*
 * compute_BayesReg:
 *
 * compute the (mle) linear parameters Bmu (the mean)
 * A (the correllation matrix) and its inverse and
 * calculate the product t(Bmu) %*% A %*% Bmu
 */

bool compute_BayesReg(unsigned int m, double **XtX, double *XtY, 
		     double *tau2i, double s2, BayesReg *breg)
{
  /* sanity checks */
  if(m == 0) return true;
  assert(m == breg->m);

  /* compute: A = XtX + Dtaui */
  for(unsigned int i=0; i<m; i++) 
    breg->A[i][i] = XtX[i][i] + tau2i[i];

  /* Ai = inv(A) */
  dup_matrix(breg->A_util, breg->A, m, m);
  id(breg->Ai,m);
  int info = linalg_dposv(m, breg->A_util, breg->Ai);
  /* now A_util is useless */
  
  /* unsuccessful inverse */
  if(info != 0) return false;
  
  /* compute: Bmu = Ai %*% Xt %*% ytilde */
  linalg_dsymv(m, 1.0, breg->Ai, m, XtY, 1, 0.0, breg->bmu, 1);

  /* t(Bmu) %*% (A) %*% Bmu */
  linalg_dsymv(m, 1.0, breg->A, m, breg->bmu, 1, 0.0, breg->ABmu, 1);
  breg->BtAB = linalg_ddot(m, breg->bmu, 1, breg->ABmu, 1);

  /* copy in the Vb matrix from Ai and s2 */
  refresh_Vb(breg, s2);

  /* sanity setting for lprob */
  breg->lprob = -1e300*1e300;

  /* return success */
  return true;
}


/*
 * refresh_Vb:
 *
 * copy breg->Ai*s2 into the breg->Vb matrix and 
 * subsequently reset the Vb_state indicator
 */

void refresh_Vb(BayesReg *breg, const double s2)
{
  /* compute: Vb = s2*Ai */
  dup_matrix(breg->Vb, breg->Ai, breg->m, breg->m);
  scalev(*(breg->Vb), breg->m * breg->m, s2);
  breg->Vb_state = COV;
}


/*
 * DrawBeta
 *
 * Gibbs draw for the beta m-vector conditional on the
 * other Bayesian lasso parameters -- assumes that updated
 * bmu and Ai have been precomputed by Compute() --
 * see the corresponding C function below
 */

void Blasso::DrawBeta(void)
{
  /* sanity checks */
  if(m == 0) return;

  draw_beta(m, breg, s2, rn, RJ);
}


/*
 * draw_beta:
 *
 * Gibbs draw for the beta m-vector conditional on the
 * other Bayesian lasso parameters -- assumes that updated
 * bmu and Ai have been precomputed by Compute() --
 * destroys Vb
 */

void draw_beta(const unsigned int m, BayesReg *breg, 
	       const double s2, double *rn, const bool lpdf)
{
  /* sanity check */
  assert(m == breg->m);
  if(m == 0) return;
  
  /* decompose Vb to have part chol and part orig triangles */
  assert(breg->Vb_state == COV);
  linalg_dpotrf(m, breg->Vb);
  
  /* record the changed Vb state */
  breg->Vb_state = CHOLCOV;
  
  /* draw */
  for(unsigned int i=0; i<m; i++) rn[i] = norm_rand();
  mvnrnd(breg->beta, breg->bmu, breg->Vb, rn, m);

  /* possibly calculate the beta (log) pdf */
  if(lpdf) beta_lprob(breg);
  else breg->lprob = -1e300*1e300;
  /* otherwise invalidate the lprob calculation */
}


/*
 * beta_lprob:
 * 
 * calculate the log density of the sample breg->beta
 * under its full conditional with parameters breg->nmu
 * and breg->Vb (which must be in its cholesky decomposed
 * state)
 */ 

double beta_lprob(BayesReg *breg)
{
  assert(breg->Vb_state == CHOLCOV);

  /* calculate the log proposal probability */
  breg->lprob = mvnpdf_log(breg->beta,breg->bmu,breg->Vb, breg->m); 

  /* record the changed Vb state */
  breg->Vb_state = NOINIT;

  return(breg->lprob);
}


/*
 * DrawS2Margin:
 *
 * Gibbs draw for the s2 scalar conditional on the
 * other Bayesian lasso parameters -- assumes that updated
 * bmu and Ai have been precomputed by Compute() -- does
 * not depend on beta because it has been integrated out
 */

void Blasso::DrawS2Margin(void)
{
  /* shape = (n-1)/2 + m/2 */
  double shape = a;
  if(reg_model == LASSO) shape += (n-1)/2.0;//+ m/2.0;
  else shape += n/2.0 - m/2.0;
  assert(shape > 0.0);
  /* THE NON-LASSO CASE MAY BE INCORRECT UNLESS TAU2I = 0 */
  
  /* rate = (X*beta - Y)' (X*beta - Y) / 2 + B'DB / 2*/
  double scale = b + (YtY - breg->BtAB)/2.0;
  
  /* draw the sample and return it */
  s2 = 1.0/rgamma(shape, 1.0/scale);

  /* check for a problem */
  if(scale <= 0) {
    PrintParams(stdout);
    myprintf(stdout, "YtY=%.20f, BtAB=%.20f\n", YtY, breg->BtAB);
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
  /* sums2 = (X*beta - Y)' (X*beta - Y); then resid not needed */
  double sums2 = sum_fv(resid, n, sq);

  /* BtDB = beta'D beta/tau2 as long as lambda != 0 */
  /* MIGHT EVENTRUALLY NEED TO ALLOW ZERO-LAMBDA WITH FIXED TAU2I */
  double BtDiB;
  if(m > 0 && reg_model == LASSO) {
    dupv(BtDi, breg->beta, m);
    scalev2(BtDi, m, tau2i);
    BtDiB = linalg_ddot(m, BtDi, 1, breg->beta, 1);
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
  if(m == 0) return;
  else if(lambda2 <= 0) { refresh_Vb(breg, s2); return; }

  /* part of the mu parameter to the inv-gauss distribution */
  l_numer = log(lambda2) + log(s2);

  for(unsigned int j=0; j<m; j++) {
      
    /* the rest of the mu parameter */
    l_mup = 0.5*l_numer - log(fabs(breg->beta[j])); 
    
    /* sample from the inv-gauss distn */
    tau2i[j] = rinvgauss(exp(l_mup), lambda2);    
    
    /* check to make sure there were no numerical problems */
    if(tau2i[j] <= 0) {
#ifdef DEBUG
      myprintf(stdout, "j=%d, m=%d, n=%d, l2=%g, s2=%g, beta=%g, tau2i=%g\n", 
	       j, m, n, lambda2, s2, breg->beta[j], tau2i[j]);
#endif
      tau2i[j] = 0;
    }
  }
  
  /* Since tau2i has changed, we need to recompute the linear
     parameters */
  assert(Compute());
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
    printMatrix(X, n, M, outfile);
  } else {
    myprintf(outfile, "X = NULL\n");
  }

  /* print the response vector */
  myprintf(outfile, "Y = ");
  printVector(Y, n, outfile, HUMAN);
}


/*
 * logPosterior:
 *
 * calculate the log posterior of the Bayesian lasso
 * model with the current parameter settings -- up to
 * an addive constant of proportionality (on the log scale)
 */

double Blasso::logPosterior(void)
{
  return log_posterior(n, m, resid, breg->beta, s2, tau2i, lambda2,
		       a, b, r, delta);
}



/*
 * log_posterior
 *
 * calculate the log posterior of the Bayesian lasso
 * model with the current parameter settings -- up to
 * an addive constant of proportionality (on the log scale)
 */

double log_posterior(const unsigned int n, const unsigned int m, 
		     double *resid, double *beta, const double s2, 
		     double *tau2i, const double lambda2, 
		     const double a, const double b, const double r,
		     const double delta)
{
  /* for summing in the (log) posterior */
  double lpost = 0.0;

  /* calculate the likelihood prod[N(resid | 0, s2)] */
  double sd = sqrt(s2);
  for(unsigned int i=0; i<n; i++) 
    lpost += dnorm(resid[i], 0.0, sd, 1);

  /* add in the prior for beta */
  for(unsigned int i=0; i<m; i++) 
    if(tau2i[i] > 0)
      lpost += dnorm(beta[i], 0.0, sd*sqrt(1.0/tau2i[i]), 1);

  /* add in the prior for s2 */
  if(a != 0 && b != 0) 
    lpost += dgamma(1.0/s2, a, 1.0/b, 1);

  /* add in the prior for tau2 */
  if(lambda2 != 0)
    for(unsigned int i=0; i<m; i++)
      lpost += dexp(1.0/tau2i[i], 2.0/lambda2, 1);

  /* add in the lambda prior */
  if(lambda2 != 0 && r != 0 && delta != 0) 
    lpost += dgamma(lambda2, r, 1.0/delta, 1);

  /* return the log posterior */
  return lpost;
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
 * add_col:
 *
 * Do the bookkeeping necessary to add the column
 * col to the design matrix -- called at the end
 * of a successful RJup move, before m is increased
 */

void Blasso::add_col(unsigned int i, unsigned int col)
{
  assert(m != M);
  assert(pb[col] == false);
  assert(pout[i] == (int) col);
  pb[col] = true;
  pin = (int*) realloc(pin, sizeof(int)*(m+1));
  pin[m] = col;
  pout[i] = pout[M-m-1];
  pout =(int*) realloc(pout, sizeof(int)*(M-m-1));
  m++;
}


/*
 * remove_col:
 *
 * Do the bookkeeping necessary to remove the column
 * col from the design matrix -- called at the end
 * of a successful RJup move, before m is decreased
 */

void Blasso::remove_col(unsigned int i, unsigned int col)
{
  assert(m != 0);
  assert(pb[col] == true);
  assert(pin[i] == (int) col);
  pb[col] = false;
  pin[i] = pin[m-1];
  pin = (int*) realloc(pin, sizeof(int)*(m-1));
  pout =(int*) realloc(pout, sizeof(int)*(M-m+1));
  pout[M-m] = col;
  m--;
}


int Blasso::Method(void)
{
  if(M == 0) return 1; /* complete */

  if(RJ) { /* reversible jump */
    
    if(reg_model == LASSO) return 2;  /* rjlasso */
    else return 3;  /* rjols */

  } else { /* no RJ */

    if(reg_model == LASSO) return 4; /* lasso */
    else return 5; /* ols */
  }
}


/*
 * mvnrnd:
 * 
 * draw from a multivariate normal mu is an n-array, 
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
 * mvnpdf_log:
 * 
 * logarithm of the density of x (n-vector) distributed 
 * multivariate normal with mean mu (n-vector) and covariance 
 * matrix cov (n x n) covariance matrix is destroyed 
 * (written over)
 */

double mvnpdf_log(double *x, double *mu, double **cov_chol, 
		  const unsigned int n)
{
  double log_det_sigma, discrim;
  double *xx;
  
  /* duplicate of the x vector */
  xx = new_dup_vector(x, n);
  
  /* det_sigma = prod(diag(R)) .^ 2 */
  log_det_sigma = log_determinant_chol(cov_chol, n);
  
  /* xx = (x - mu) / R; */
  linalg_daxpy(n, -1.0, mu, 1, xx, 1);
  /*linalg_dtrsv(CblasTrans,n,cov,n,xx,1);*/
  linalg_dtrsv(CblasTrans,n,cov_chol,n,xx,1);
  
  /* discrim = sum(x .* x, 2); */
  /* discrim = linalg_ddot(n, xx, 1, xx, 1); */
  discrim = linalg_ddot(n, xx, 1, xx, 1);
  free(xx);
  
  /*myprintf(stderr, "discrim = %g, log(deg_sigma) = %g\n", discrim, log_det_sigma);*/
  return -0.5 * (discrim + log_det_sigma) - n*M_LN_SQRT_2PI;
}


/*
 * log_determinant_chol:
 * 
 * returns the log determinant of the n x n
 * choleski decomposition of a matrix M
 */

double log_determinant_chol(double **M, const unsigned int n)
{
  double log_det;
  unsigned int i;
  
  /* det = prod(diag(R)) .^ 2 */
  log_det = 0;
  for(i=0; i<n; i++) log_det += log(M[i][i]);
  log_det = 2*log_det;
  
  return log_det;
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


double mh_accep_ratio(unsigned int n, double *resid, double *x, double bnew, 
		      double t2i, double mub, double vb, double s2)
{
  /* calculate the sum of squared errors */
  double e2 = 0.0;
  for(unsigned int i=0; i<n; i++) e2 += sq(resid[i] - bnew*x[i]);
  e2 = 0.0-1.0/(2.0*s2);
  
  /* prior for beta */
  double pb = 0.0-sq(bnew)*t2i/(2.0*s2);

  /* proposal for beta */
  double qb = 0.5*sq(bnew - mub)/vb;

  /* log of the exponent in the acceptance ratio */
  double lepo = e2 + pb + qb;

  /* now the non-exponent stuff */
  double lnepo = 0.5*(vb - 1.0/t2i);

  /* exponentiate and return */
  return(exp(lepo + lnepo));
}


extern "C"
{
/*
 * lasso_draw_R
 *
 * function currently used for testing the above functions
 * using R input and output
 */

void blasso_R(int *T, int *thin, int *M, int *n, double *X_in, 
	      double *Y, double *lambda2, double *mu, int *RJ, 
	      int *Mmax, double *beta, int *m, double *s2, 
	      double *tau2i, double *lpost, double *r, double *delta, 
	      double *a, double *b, int *rao_s2, int *normalize, int *verb)
{
  double **X, **beta_mat, **tau2i_mat;
  int i;

  assert(*T > 1);

  /* copy the vector input X into matrix form */
  X = (double **)  malloc(sizeof(double*) * (*n));
  X[0] = X_in;
  for(i=1; i<(*n); i++) X[i] = X[i-1] + (*M);

  /* get the random number generator state from R */
  GetRNGstate();

  /* initialize a matrix for beta samples */
  beta_mat = (double **) malloc(sizeof(double*) * (*T));
  beta_mat[0] = beta;
  for(i=1; i<(*T); i++) beta_mat[i] = beta_mat[i-1] + (*M);

  /* initialize a matrix for tau2i samples */
  tau2i_mat = (double **)  malloc(sizeof(double*) * (*T));
  tau2i_mat[0] = tau2i;
  for(i=1; i<(*T); i++) tau2i_mat[i] = tau2i_mat[i-1] + (*M);

  /* create a new Bayesian lasso regression */
  Blasso *blasso = 
    new Blasso(*M, *n, X, Y, (bool) *RJ, *Mmax, beta_mat[0], 
	       lambda2[0], s2[0], tau2i_mat[0], *r, *delta, *a, 
	       *b, (bool) *rao_s2, (bool) *normalize, *verb);

  /* Gibbs draws for the parameters */
  blasso->Rounds((*T)-1, *thin, &(lambda2[1]), &(mu[1]), &(beta_mat[1]), 
		 &(m[1]), &(s2[1]), &(tau2i_mat[1]), &(lpost[1]));

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


/*
 * mvnpdf_log_R:
 *
 * for testing the mvnpdf_log function 
 */

void mvnpdf_log_R(double *x, double *mu, double *cov, int *n, double *out)
{
  double **covM = new_matrix(*n, *n);
  dupv(covM[0], cov, (*n)*(*n));
  linalg_dpotrf(*n, covM);
  *out = mvnpdf_log(x, mu, covM, *n);
  delete_matrix(covM);
}
}
