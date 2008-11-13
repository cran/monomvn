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

// #define DEBUG

/*
 * Blasso:
 *
 * the typical constructor function that initializes the
 * parameters to default values, and does all pre-processing
 * necesary to start sampling from the posterior distribution
 * of the Bayesian Lasso parameters -- called by Bmonomvn
 */

Blasso::Blasso(const unsigned int M, const unsigned int n, double **Xorig, 
	       double *Xnorm, const double Xnorm_scale, double *Xmean, 
	       const unsigned int ldx, double *Y, const bool RJ, 
	       const unsigned int Mmax, double *beta_start, const double s2_start, 
	       const double lambda2_start, const double mprior, 
	       const double r, const double delta, const REG_MODEL reg_model, 
	       const bool rao_s2, const unsigned int verb)
{
  /* sanity checks */
  if(Mmax >= n) assert(RJ || reg_model == LASSO || reg_model == RIDGE);

  /* copy RJ setting */
  this->RJ = RJ;
  this->tau2i = this->beta = this->rn = this->BtDi = NULL;
  this->lpost = -1e300*1e300;

  /* initialize the active set of columns of X */
  InitIndicators(M, Mmax, beta_start);

  /* copy the Rao-Blackwell option for s2 */
  this->rao_s2 = rao_s2;

  /* initialize the input data */ 
  InitXY(n, Xorig, Xnorm, Xnorm_scale, Xmean, ldx, Y, true);

  /* copy verbosity argument */
  this->verb = verb;

  /* initialize the mprior value */
  this->mprior = mprior;

  /* initialize the parameters */
  this->r = r;
  this->delta = delta;

  /* this function will be modified to depend on whether OLS 
     must become lasso */
  InitParams(reg_model, beta_start, s2_start, lambda2_start);

  /* initialize the residula vector */
  resid = new_dup_vector(Y, n);

  /* only used by one ::Draw function */
  Xbeta_v = new_vector(n);

  /* must call ::Init function first thing */
  breg = NULL;
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
	       double *beta, const double lambda2, const double s2, 
	       double *tau2i, const double mprior, const double r, 
	       const double delta, const double a, const double b, 
	       const bool rao_s2, const bool normalize, 
	       const unsigned int verb)
{
  /* copy RJ setting */
  this->RJ = RJ;
  this->tau2i = this->beta = this->rn = this->BtDi = NULL;
  this->lpost = -1e300*1e300;

  /* initialize the active set of columns of X */
  InitIndicators(M, Mmax, beta);

  /* copy the Rao-Blackwell option for s2 */
  this->rao_s2 = rao_s2;

  /* initialize the input data */ 
  InitXY(n, X, Y, normalize);

  /* copy verbosity argument */
  this->verb = verb;

  /* initialize the mprior value */
  this->mprior = mprior;

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

  /* this utility vector is not used here */
  Xbeta_v = NULL;

  /* must call ::Init function first thing */
  breg = NULL;
 }


/*
 * Init:
 *
 * part of the constructor that could result in an error
 * so its contents are moved outside in case such an event
 * happens which would always cause a segmentation fault
 */

void Blasso::Init()
{
  /* sanity check */
  assert(breg == NULL);

  /* initialize the regression utility */
  InitRegress();

 /* calculate the initial linear mean and corellation matrix */
  if(!Compute()) error("ill-posed regression in Init");
}


/*
 * InitIndicators:
 *
 * set the total number of colums, M, and then set the
 * initial number of non-zero columns, m
 */

void Blasso::InitIndicators(const unsigned int M, unsigned int Mmax, 
			    double *beta)
{
  /* copy the dimension parameters*/
  this->M = M;
  this->Mmax = Mmax;

  /* sanity checks */
  assert(Mmax <= M);
  if(!RJ) assert(Mmax == M);

  /* find out which betas are non-zero, thus setting m */
  pb = (bool*) malloc(sizeof(bool) * M);

  /* check if model initialization is specified by 
     an initial beta vector */
  if(beta != NULL) {

    /* fill in the indicators to true where beta[i] != NULL */
    m = 0;
    for(unsigned int i=0; i<M; i++) {
      if(beta[i] != 0) { pb[i] = true; m++; }
      else pb[i] = false;
      assert(m <= Mmax);
    }

    /* see if we are starting in a non-saturated model when RJ
       is false, and warn if so */
    if(!RJ && m < M) 
      warning("RJ=FALSE, but not in saturated model (m=%d, M=%d), try RJ=\"p\"",
	      m, M);

  } else { /* otherwise default depends on RJ */

    /* if using RJ, then start nearly saturated */
    if(RJ) m = (unsigned int) (0.9* Mmax);
    else m = Mmax;  /* otherwise start saturated */

    /* fill in the corresponding booleans */
    for(unsigned int i=0; i<m; i++) pb[i] = true;
    for(unsigned int i=m; i<M; i++) pb[i] = false;
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
 * initialize the input data (X and Y) by allocating new
 * memory and calculating normalized stuff
 */

void Blasso::InitXY(const unsigned int n, double **Xorig, double *Y, 
		    const bool normalize)
{
  this->ldx = M;
  this->copies = true;
  this->n = n;

  /* copy the input matrix */
  this->Xorig = new_dup_matrix(Xorig, n, M);

  /* calculate the mean of each column of X*/
  Xmean = new_zero_vector(M);
  wmean_of_columns(Xmean, this->Xorig, n, M, NULL);
  
  /* center X */
  double **X = new_dup_matrix(Xorig, n, M);
  center_columns(X, Xmean, n, M);
  
  /* normalize X, like Efron & Hastie */
  /* (presumably so that [good] lambda doesn't increase with n ??) */
  this->Xnorm_scale = 1.0;
  this->normalize = normalize;
  if(this->normalize) {
    Xnorm = new_zero_vector(M);
    sum_of_columns_f(Xnorm, X, n, M, sq);
    for(unsigned int i=0; i<M; i++) Xnorm[i] = sqrt(Xnorm[i]);
    norm_columns(X, Xnorm, n, M);
  } else Xnorm = NULL;

  /* extract the active columns of X */
  Xp = new_p_submatrix(pin, X, n, m);
  delete_matrix(X);

  /* now handle everything that has to do with Y and Xp */
  InitY(n, Y);
}


/* 
 * InitXY:
 *
 * initialize the input data (X and Y) from pointers allocated
 * outside this module (likely in Bmonomvn::)
 */

void Blasso::InitXY(const unsigned int n, double **Xorig, double *Xnorm,
		    const double Xnorm_scale, double *Xmean, 
		    const unsigned int ldx, double *Y, const bool normalize)
{
  this->copies = false;
  this->n = n;

  /* copy the POINTER to the input matrices */
  /* NOTE THAT Xorig AND X WILL HAVE LARGER LEADING DIMENSIONS */
  this->Xorig = Xorig;
  this->Xmean = Xmean;
  this->normalize = normalize;
  this->Xnorm = Xnorm;
  this->Xnorm_scale = Xnorm_scale;
  this->ldx = ldx;

  /* extract the active columns of X */
  Xp = new_p_submatrix(pin, Xorig, n, m);
  for(unsigned int i=0; i<n; i++) {
    for(unsigned int j=0; j<m; j++) {
      Xp[i][j] -= Xmean[pin[j]];
      if(normalize) Xp[i][j] /= Xnorm_scale * Xnorm[pin[j]];
    }
  }

  /* now handle everything that has to do with Y and Xp */
  InitY(n, Y);
}


/* 
 * InitY:
 *
 * handle the rest of the XY initialization that has to do with
 * Y -- must happen after all X initialization is done 
 */

void Blasso::InitY(const unsigned int n, double *Y)
{	   
  /* sanity check */
  assert(this->n == n);

  /* center Y */
  this->Y = new_dup_vector(Y, n);
  Ymean = meanv(Y, n);
  centerv(this->Y, n, Ymean); 

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
  assert(breg == NULL);
  breg = new_BayesReg(m, n, Xp);

  /* allocate the other miscellaneous vectors needed for
     doing the regressions */
  assert(BtDi == NULL && rn == NULL);
  if(!rao_s2) BtDi = new_vector(m);
  else BtDi = NULL;
  rn = new_vector(M);
}


/*
 * new_BayesReg:
 *
 * allocate a new regression utility structure
 */

BayesReg* new_BayesReg(const unsigned int m, const unsigned int n, 
		       double **Xp)
{
  /* allocate the structure */
  BayesReg *breg = (BayesReg*) malloc(sizeof(struct bayesreg));
  breg->m = m;
  
  /* fill A with t(X) %*% X -- i.e. XtX */
  breg->A = new_zero_matrix(m, m);
  if(breg->A) linalg_dgemm(CblasNoTrans,CblasTrans,m,m,n,1.0,
		       Xp,m,Xp,m,0.0,breg->A,m);

  /* save the diagonal of XtX in a separate vector */
  breg->XtX_diag = new_vector(m);
  for(unsigned int i=0; i<m; i++) breg->XtX_diag[i] = breg->A[i][i];
  
  alloc_rest_BayesReg(breg);
  return(breg);
}


/*
 * alloc_rest_BayesReg:
 *
 * allocate memory for the rest of the entires of the 
 * BayesReg structure, without initialization
 */

void alloc_rest_BayesReg(BayesReg* breg)
{
  /* utility for Gibbs sampling parameters for beta */
  breg->A_chol = new_matrix(breg->m, breg->m);
  breg->Ai = new_matrix(breg->m, breg->m);
  breg->ABmu = new_vector(breg->m);
  breg->BtAB = 0.0;
  breg->ldet_Ai = 0.0;

  /* allocate the Gibbs sampling parameters for beta */
  breg->bmu = new_vector(breg->m);  
  breg->Vb = new_matrix(breg->m, breg->m);
  breg->Vb_state = NOINIT;
}


/*
 * plus1_BayesReg:
 *
 * allocate a BayesReg structure derived from an old one with
 * one new column, xnew, in the design matrix.  Xp is the 
 * pointer to the m old columns -- the new one has m+1
 */

BayesReg* plus1_BayesReg(const unsigned int m, const unsigned int n,
			  BayesReg *old, double *xnew, double **Xp)
{
  /* sanity check */
  assert(m == old->m);

  /* allocate the structure */
  BayesReg *breg = (BayesReg*) malloc(sizeof(struct bayesreg));
  breg->m = m+1;

  /* fill A with t(X) %*% X -- i.e. XtX */
  breg->A = new_matrix(m+1, m+1);
  dup_matrix(breg->A, old->A, m, m);

  /* XtX[m][m] = t(x) %*% x */
  breg->A[m][m] = linalg_ddot(n, xnew, 1, xnew, 1);

  /* get the new row of XtX_new[col,] = X[,col] %*% X[,1:m] */
  for(unsigned int j=0; j<m; j++) {
    breg->A[m][j] = 0;
    for(unsigned int i=0; i<n; i++) breg->A[m][j] += Xp[i][j]*xnew[i];
  }

  /* copy the rows to the columns */
  dup_col(breg->A, m, breg->A[m], m);

  /* save the diagonal of XtX in a separate vector */
  breg->XtX_diag = new_vector(m+1);
  dupv(breg->XtX_diag, old->XtX_diag, m);
  breg->XtX_diag[m] = breg->A[m][m];

  /* finish un-initialized of the rest of the entries and return */
  alloc_rest_BayesReg(breg);
  return(breg);
}


/*
 * InitParams:
 *
 * pick (automatic) starting values for the parameters based on the type
 * of model
 */

void Blasso::InitParams(const REG_MODEL reg_model, double *beta, double s2,
			double lambda2)
{
  /* sanity check */
  assert(this->tau2i == NULL && this->beta == NULL);
  this->reg_model = reg_model;

  /* set the LASSO & RIDGE lambda2 and tau2i initial values */
  if(reg_model != OLS) {

    /* assign starting lambda2 value */
    this->lambda2 = lambda2;
    if(m > 0 && lambda2 <= 0  && reg_model == LASSO) {
      warning("starting lambda2 (%g) <= 0 is invalid (m=%d, M=%d)", 
	      lambda2, m, M);
      this->lambda2 = 1.0;
    } else this->lambda2 = lambda2;

    /* tau2i is only used by LASSO, not RIDGE */
    if(reg_model == LASSO) tau2i = ones(m, 1.0);
    else { /* for RIDGE */
       tau2i = NULL;
       if(m == 0) this->lambda2 = 0.0;
    }

  } else { /* OLS */
    if(lambda2 != 0)
      warning("starting lambda2 value (%g) must be zero (m=%d, M=%d)", 
	      lambda2, m, M);
    this->lambda2 = 0.0;
    tau2i = NULL;
  }

  /* allocate beta */
  this->beta = new_vector(m);

  /* initialize the initial beta vector */
  if(beta) {
    /* norm the beta samples, like Efron and Hastie */
    if(normalize && this->m > 0) {
      scalev2(beta, M, Xnorm);
      scalev(beta, M, Xnorm_scale);
    }
    
    /* copy in the beta vector */
    copy_sub_vector(this->beta, pin, beta, m);
  } else {
    /* initialize beta so that it is non-zero */
    for(unsigned int i=0; i<m; i++) this->beta[i] = 1.0;
  }

  /* initialize regression coefficients */
  this->s2 = s2;

  /* default setting when not big-p-small n */
  a = b = 0;

  /* need to help with a stronger s2-prior beta parameter when p>n */
  if(reg_model != OLS && !RJ && M >= n) {
      a = 3.0/2.0;
      b = Igamma_inv(a, 0.95*gammafn(a), 0, 0)*YtY;
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

  /* copy in the tau2i vector */
  assert(this->tau2i == NULL);
  if(tau2i != NULL) this->tau2i = new_sub_vector(pin, tau2i, m);

  /* allocate beta */
  this->beta = new_vector(m);

  /* norm the beta samples, like Efron and Hastie */
  if(normalize && this->m > 0) {
    scalev2(beta, M, Xnorm);
    scalev(beta, M, Xnorm_scale);
  }

  /* copy in the beta vector */
  copy_sub_vector(this->beta, pin, beta, m);

  /* determine the resulting regression model */
  if(lambda2 == 0) {
    reg_model = OLS;
    assert(tau2i == NULL);
  } else if(tau2i == NULL) reg_model = RIDGE;
  else {
    if(m > 0) assert(sumv(this->tau2i, m) != 0);
    reg_model = LASSO;
  }

  /* set lambda2 to zero if m == 0  && RIDGE */
  if(M == 0 || (reg_model == RIDGE && m == 0)) this->lambda2 = 0.0;
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
  if(beta) free(beta);
  
  /* possibly copied InitXY stuff */
  if(copies) {
    if(Xorig) delete_matrix(Xorig);
    if(Xmean) free(Xmean);
    if(normalize && Xnorm) free(Xnorm);
  }

  /* other InitXY stuff */
  if(Xp) delete_matrix(Xp);
  if(XtY) free(XtY);
  if(Y) free(Y);
  if(resid) free(resid);

  /* free the regression utility */
  Economize();

  /* free extra regression utility vectors */
  if(Xbeta_v) free(Xbeta_v);

  /* free the boolean column indicators */
  if(pb) free(pb);
  if(pin) free(pin);
  if(pout) free(pout);
}


/*
 * Economize:
 *
 * frees everything that is allocated by the
 * Init function (which includes BayesReg and 
 * Init Regress)
 */

void Blasso::Economize(void)
{
  /* free the regression utility */
  if(breg) { delete_BayesReg(breg); breg = NULL; }

  /* other utility vectors */
  if(BtDi) { free(BtDi); BtDi = NULL; }
  if(rn) { free(rn); rn = NULL; }
}


/*
 * delete_BayesReg:
 *
 * free the space used by the regression utility
 * structure
 */

void delete_BayesReg(BayesReg* breg)
{
  if(breg->XtX_diag) free(breg->XtX_diag);
  if(breg->A) delete_matrix(breg->A);
  if(breg->A_chol) delete_matrix(breg->A_chol);
  if(breg->Ai) delete_matrix(breg->Ai);
  if(breg->ABmu) free(breg->ABmu);
  if(breg->bmu) free(breg->bmu);
  if(breg->Vb) delete_matrix(breg->Vb);
  // if(breg->beta) free(breg->beta);
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
  if(this->m > 0) copy_p_vector(beta, pin, this->beta, this->m);
  *s2 = this->s2;
  if(tau2i) {
    for(unsigned int i=0; i<M; i++) tau2i[i] = -1.0;
    if(this->m > 0 && this->tau2i)
      copy_p_vector(tau2i, pin, this->tau2i, this->m);
  }
  if(lambda2) *lambda2 = this->lambda2;
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
  if(tau2i) {
    myprintf(outfile, "tau2i = ");
    printVector(tau2i, m, outfile, HUMAN);
  }
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
  /* sanity check */
  assert(breg);

  /* for helping with periodic interrupts */
  time_t itime = time(NULL);

  /* assume that the initial values reside in position 0 */
  /* do T-1 MCMC rounds */
  for(unsigned int t=0; t<T; t++) {
    
    /* do thin number of MCMC draws */
    Draw(thin);

    /* if LASSO then get t-th tau2i */
    double *tau2i_samp = NULL;
    if(tau2i) { assert(reg_model == LASSO); tau2i_samp = tau2i[t]; }

    /* if LASSO or ridge */
    double *lambda2_samp = NULL;
    if(lambda2) { assert(m == 0 || reg_model != OLS); lambda2_samp = &(lambda2[t]); } 

    /* copy the sampled parameters */
    GetParams(beta[t], &(m[t]), &(s2[t]), tau2i_samp, lambda2_samp);
    
    /* get the log posterior */
    lpost[t] = this->lpost;

    /* print progress meter */
    if(verb && t > 0 && ((t+1) % 100 == 0))
      myprintf(stdout, "t=%d, m=%d\n", t+1, this->m);

    /* periodically check R for interrupts and flush console every second */
    itime = my_r_process_events(itime);
  }

  /* (un)-norm the beta samples, like Efron and Hastie */
  if(normalize) {
    norm_columns(beta, Xnorm, T, M);
    scalev(beta[0], T*M, 1.0/Xnorm_scale);
  }

  /* calculate mu samples */

  /* Xbeta = X %*% t(beta), in col-major representation */
  double **Xbeta = new_zero_matrix(T,n);
  linalg_dgemm(CblasTrans,CblasNoTrans,n,T,M,1.0,Xorig,ldx,beta,M,0.0,Xbeta,n);

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
    if(reg_model == LASSO) DrawTau2i();
    else assert(tau2i == NULL);
    
    /* only depends on tau2i for LASSO and beta for RIDGE */
    if(reg_model != OLS) DrawLambda2();
    else { /* is OLS */
      assert(lambda2 == 0 && tau2i == NULL);
      if(m > 0) refresh_Vb(breg, s2);
    }

    /* depends on pre-calculated bmu, Vb, etc, which depends
       on tau2i -- breg->Vb then becomes decomposed */
    DrawBeta();

    /* resid = X*beta - Y */
    dupv(resid, Y, n);
    if(m > 0) linalg_dgemv(CblasTrans,m,n,-1.0,Xp,m,beta,1,1.0,resid,1);

    /* choose the type of s2 GS update */
    if(rao_s2) {
      DrawS2Margin();  /* depends on bmu and Vb but not beta */
    } else DrawS2();  /* depends on beta */

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
 * copy them out to the pointers/memory passed in -- thin has been
 * adapted to be thin*Mmax;
 */

void Blasso::Draw(const unsigned int thin, double *lambda2, double *mu, 
		  double *beta, int *m, double *s2, double *tau2i, 
		  double *lpost)
{
  /* sanity check */
  assert(breg && Xbeta_v);

  /* do thin number of MCMC draws */
  Draw(Thin(thin));

  /* copy the sampled parameters */
  GetParams(beta, m, s2, tau2i, lambda2);

  /* (un)-norm the beta samples, like Efron and Hastie */
  if(normalize && this->m > 0) {
    normv(beta, M, Xnorm);
    scalev(beta, M, 1.0/Xnorm_scale);
  }

  if(this->m > 0) {
    /* Xbeta = X %*% beta, in col-major representation */
    linalg_dgemv(CblasTrans,M,n,1.0,Xorig,ldx,beta,1,0.0,Xbeta_v,1);
    
    /* mu = mean(Xbeta) */
    *mu = meanv(Xbeta_v, n);
  } else *mu = 0;
  
  /* mu = rnorm(Ymean, sqrt(s2/n)) - mean(Xbeta) */
  double mu_adj = Ymean;
  double sd = 0;
  if(thin > 0) { /* no random draws if thin > 0 */
    sd = sqrt((*s2)/n);
    mu_adj = rnorm(Ymean, sd);
  }
  *mu = mu_adj - (*mu);

  /* calculate the log posterior */
  *lpost = this->lpost;
  if(thin > 0) *lpost += dnorm(mu_adj, Ymean, sd, 1);
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
 * try to add another column to the model in the style
 * of Throughton & Godsill proposals that integrate out
 * beta
 */

void Blasso::RJup(double qratio)
{
  /* sanity checks */
  assert(m != M);

  /* randomly select a column for addition */
  int iout = (int) ((M-m) * unif_rand());
  int col = pout[iout];
  double *xnew = new_vector(n);
  for(unsigned int i=0; i<n; i++) xnew[i] = (Xorig[i][col] - Xmean[col]);
  if(normalize) for(unsigned int i=0; i<n; i++) xnew[i] /= Xnorm_scale * Xnorm[col];
  qratio *= ((double)(M-m))/(m+1);

  /* randomly propose a new tau2i component or lambda depending on reg_model */
  double prop = 1.0;
  if(reg_model == LASSO) { /* randomly propose a new tau2i-component */
    tau2i = (double*) realloc(tau2i, sizeof(double)*(m+1));
    prop = rexp(2.0/lambda2);
    tau2i[m] = 1.0 / prop;
  } else if(reg_model == RIDGE && m == 0) { /* randomly propose a new lambda */
    lambda2 = prop = rexp(1);
  } else if(reg_model == RIDGE) prop = lambda2;

  /* add a new component to XtY */
  XtY = (double*) realloc(XtY, sizeof(double)*(m+1));
  XtY[m] = linalg_ddot(n, xnew, 1, Y, 1);

  /* allocate new regression stuff */
  /* diagonal of A is taken care of inside of compute_BayesReg() */
  BayesReg *breg_new = plus1_BayesReg(m, n, breg, xnew, Xp);

  /* compute the new regression quantities */
  assert(compute_BayesReg(m+1, XtY, tau2i, lambda2, s2, breg_new));

  /* calculate the acceptance probability breg -> breg_new */
  double lalpha = rj_betas_lratio(breg, breg_new, s2, prop);
  // myprintf(stdout, "RJ-up lalpha = %g\n", lalpha);

  /* add in the forwards probabilities */
  if(reg_model == LASSO) lalpha -= dexp(prop,2.0/lambda2,1);
  else if(reg_model == RIDGE) lalpha -= dexp(lambda2, 1, 1);

  /* add in the (log) prior model probabilities */
  lalpha += lprior_model(m+1, Mmax, mprior) - lprior_model(m, Mmax, mprior);

  /* MH accept or reject */
  if(unif_rand() < exp(lalpha)*qratio) { /* accept */

    /* copy the new regression utility */
    delete_BayesReg(breg); breg = breg_new;
    
    /* draw the new beta vector */
    beta = (double*) realloc(beta, sizeof(double)*(m+1));
    draw_beta(m+1, beta, breg, s2, rn);

    /* calculate new residual vector */
    dupv(resid, Y, n);
    if(m > 0) linalg_dgemv(CblasTrans,m,n,-1.0,Xp,m,beta,1,1.0,resid,1);
    linalg_daxpy(n, 0.0 - beta[m], xnew, 1, resid, 1);

    /* other copies */
    if(BtDi) BtDi = (double*) realloc(BtDi, sizeof(double) * (m+1));

    /* add another column to the effective design matrix */
    Xp = new_bigger_matrix(Xp, n, m, n, m+1);
    dup_col(Xp, m, xnew, n);
    add_col(iout, col);

    /* calculate the new log_posterior */
    lpost = logPosterior();
    
    // myprintf(stdout, "accepted RJ-up col=%d, alpha=%g\n", col, exp(lalpha));

  } else { /* reject */
    
    /* realloc vectors */
    if(reg_model == LASSO) tau2i = (double*) realloc(tau2i, sizeof(double)*m);
    else if(reg_model == RIDGE && m == 0) lambda2 = 0;
    XtY = (double*) realloc(XtY, sizeof(double)*m);
    
    /* free new regression utility */
    delete_BayesReg(breg_new);

    // myprintf(stdout, "rejected RJ-up col=%d, alpha=%g\n", col, exp(lalpha));
  }

  /* clean up */
  free(xnew);
}


/*
 * RJdown:
 *
 * try to remove a column to the model in the style
 * of Throughton & Godsill proposals that integrate out
 * beta
 */

void Blasso::RJdown(double qratio)
{
 /* sanity checks */
  assert(m != 0);

  /* select a column for deletion */
  int iin = (int) (m * unif_rand());
  int col = pin[iin];
  qratio *= ((double)m)/(M-m+1);

  /* make the new design matrix with one fewer column */
  double **Xp_new = new_dup_matrix(Xp, n, m-1);
  if(iin != ((int)m)-1)
    for(unsigned int i=0; i<n; i++) Xp_new[i][iin] = Xp[i][m-1];
  
  /* select corresponding tau2i-component */
  double prop = 1.0;
  if(reg_model == LASSO) {
    prop = 1.0/tau2i[iin];
    tau2i[iin] = tau2i[m-1];
    tau2i = (double*) realloc(tau2i, sizeof(double)*(m-1));
  } else if(reg_model == RIDGE && m == 1) { 
    prop = lambda2;
    lambda2 = 0.0;
  } else if(reg_model == RIDGE) prop = lambda2;
  
  /* remove component of XtY */
  double xty = XtY[iin];
  XtY[iin] = XtY[m-1];
  XtY = (double*) realloc(XtY, sizeof(double)*(m-1));

  /* allocate new regression stuff */
  BayesReg *breg_new = new_BayesReg(m-1, n, Xp_new);

  /* compute the new regression quantities */
  bool success = compute_BayesReg(m-1, XtY, tau2i, lambda2, s2, breg_new);
  assert(success);

  /* calculate the acceptance probability breg -> breg_new */
  double lalpha = rj_betas_lratio(breg, breg_new, s2, prop);
  // myprintf(stdout, "RJ-down lalpha = %g\n", lalpha);
  
  /* add in the backwards probabilities */
  if(reg_model == LASSO) lalpha += dexp(prop,2.0/lambda2,1);
  else if(reg_model == RIDGE && m == 1) lalpha += dexp(prop,1,1);

  /* add in the (log) prior model probabilities */
  lalpha += lprior_model(m-1, Mmax, mprior) - lprior_model(m, Mmax, mprior);

  /* MH accept or reject */
  if(unif_rand() < exp(lalpha)*qratio) { /* accept */

    /* myprintf(stdout, "accept RJ-down, col=%d, lnew=%g, lold=%g, pr=%g, A=%g\n", 
       col, lpost_new, lpost, exp(lpost_new - lpost), exp(lalpha)); */

    /* copy the new regression utility */
    delete_BayesReg(breg); breg = breg_new;

    /* draw the new beta vector */
    beta = (double*) realloc(beta, sizeof(double)*(m-1));
    draw_beta(m-1, beta, breg, s2, rn);
    
    /* calculate new residual vector */
    dupv(resid, Y, n);
    if(m-1 > 0) linalg_dgemv(CblasTrans,m-1,n,-1.0,Xp_new,m-1,
			     beta,1,1.0,resid,1);

    /* other */
    if(BtDi) BtDi = (double*) realloc(BtDi, sizeof(double) * (m-1));

    /* permanently remove the column to the effective design matrix */
    delete_matrix(Xp); Xp = Xp_new;
    remove_col(iin, col);

    /* calculate the new log_posterior */
    lpost = logPosterior();

  } else { /* reject */

    /* realloc vectors */
    if(reg_model == LASSO) { 
      tau2i = (double*) realloc(tau2i, sizeof(double)*m);
      tau2i[m-1] = tau2i[iin]; tau2i[iin] = 1.0/prop;
    } else if(reg_model == RIDGE && m == 1) lambda2 = prop;
    XtY = (double*) realloc(XtY, sizeof(double)*m);
    XtY[m-1] = XtY[iin]; XtY[iin] = xty;
    
    /* free new regression utility */
    delete_BayesReg(breg_new);
    delete_matrix(Xp_new);

    /* myprintf(stdout, "reject RJ-down, col=%d, lnew=%g, lold=%g, qrat=%g, A=%g\n", 
       col, lpost_new, lpost, qratio, exp(lalpha)); */
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

  bool ret = compute_BayesReg(m, XtY, tau2i, lambda2, s2, breg); 

  return ret && (YtY - breg->BtAB > 0);
}


/*
 * compute_BayesReg:
 *
 * compute the (mle) linear parameters Bmu (the mean)
 * A (the correllation matrix) and its inverse and
 * calculate the product t(Bmu) %*% A %*% Bmu
 */

bool compute_BayesReg(const unsigned int m, double *XtY, double *tau2i, 
		      const double lambda2, const double s2, BayesReg *breg)
{
  /* sanity checks */
  if(m == 0) return true;
  assert(m == breg->m);

  /* compute: A = XtX + Dtaui */
  if(tau2i) {
    for(unsigned int i=0; i<m; i++) 
      breg->A[i][i] = breg->XtX_diag[i] + tau2i[i];
  } else if(lambda2 != 0) {
    for(unsigned int i=0; i<m; i++) 
      breg->A[i][i] = breg->XtX_diag[i] + 1.0/lambda2;
  }
  
  /* Ai = inv(A) */
  dup_matrix(breg->A_chol, breg->A, m, m);
  id(breg->Ai,m);
  int info = linalg_dposv(m, breg->A_chol, breg->Ai);
  /* now A_chol = chol(A) */
  
  /* unsuccessful inverse */
  if(info != 0) return false;

  /* compute: ldet_Ai = log(det(Ai)) */
  breg->ldet_Ai = 0.0 - log_determinant_chol(breg->A_chol, m);

  /* compute: Bmu = Ai %*% Xt %*% ytilde */
  linalg_dsymv(m, 1.0, breg->Ai, m, XtY, 1, 0.0, breg->bmu, 1);

  /* compute: BtAB = t(Bmu) %*% (A) %*% Bmu */
  linalg_dsymv(m, 1.0, breg->A, m, breg->bmu, 1, 0.0, breg->ABmu, 1);
  breg->BtAB = linalg_ddot(m, breg->bmu, 1, breg->ABmu, 1);

  /* copy in the Vb matrix from Ai and s2 */
  refresh_Vb(breg, s2);

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
  assert(breg->m > 0);
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

  draw_beta(m, beta, breg, s2, rn);
}


/*
 * draw_beta:
 *
 * Gibbs draw for the beta m-vector conditional on the
 * other Bayesian lasso parameters -- assumes that updated
 * bmu and Ai have been precomputed by Compute() --
 * destroys Vb
 */

void draw_beta(const unsigned int m, double *beta, BayesReg *breg, 
	       const double s2, double *rn)
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
  mvnrnd(beta, breg->bmu, breg->Vb, rn, m);
}


/*
 * beta_lprob:
 * 
 * calculate the log density of the sample beta
 * under its full conditional with parameters breg->nmu
 * and breg->Vb (which must be in its cholesky decomposed
 * state)
 */ 

double beta_lprob(const unsigned int m, double *beta, BayesReg *breg)
{
  assert(m == breg->m);
  assert(breg->Vb_state == CHOLCOV);

  /* calculate the log proposal probability */
  double lprob = mvnpdf_log(beta, breg->bmu, breg->Vb, breg->m); 

  /* record the changed Vb state */
  breg->Vb_state = NOINIT;

  return lprob;
}


/*
 * rj_betas_lratio:
 *
 * calculate the (integrated) acceptance ratio for RJ moves
 * as adapted from the Throughton and Godsill equations 
 * relating to AR models
 *
 */

double rj_betas_lratio(BayesReg *bold, BayesReg *bnew, 
		       const double s2, const double tau2)
{
  int mdiff = bnew->m - bold->m;
  assert(abs(mdiff) == 1);
  double lratio = 0.5*(bnew->ldet_Ai - bold->ldet_Ai);
  lratio += 0.5*(bnew->BtAB - bold->BtAB)/s2;
  lratio -= 0.5*mdiff*(log(s2) + log(tau2));
  return(lratio);
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
  if(reg_model != OLS) shape += (n-1)/2.0;//+ m/2.0;
  else shape += n/2.0 - m/2.0;
  assert(shape > 0.0);
  
  /* rate = (X*beta - Y)' (X*beta - Y) / 2 + B'DB / 2*/
  double scale = b + (YtY - breg->BtAB)/2.0;
  
  /* draw the sample and return it */
  s2 = 1.0/rgamma(shape, 1.0/scale);
  
  /* check for a problem */
  if(scale <= 0) error("ill-posed regression in DrawS2, scale <= 0");
}


/*
 * DrawS2:
 *
 * Gibbs draw for the s2 scalar conditional on the
 * other Bayesian lasso parameters, depends on beta
 */

void Blasso::DrawS2(void)
{
  /* sums2 = (X*beta - Y)' (X*beta - Y); */
  double sums2 = sum_fv(resid, n, sq);

  /* BtDB = beta'D beta/tau2 as long as lambda != 0 */
  double BtDiB;
  if(m > 0 && reg_model == LASSO) {
    dupv(BtDi, beta, m);
    if(tau2i) scalev2(BtDi, m, tau2i);
    else scalev(BtDi, m, 1.0/lambda2);
    BtDiB = linalg_ddot(m, BtDi, 1, beta, 1);
  } else BtDiB = 0.0;
    
  /* shape = (n-1)/2 + m/2 */
  double shape = a;
  if(reg_model != OLS) shape += (n-1)/2.0 + m/2.0;
  else shape += (n-1)/2.0; // - m/2.0;
  
  /* rate = (X*beta - Y)' (X*beta - Y) / 2 + B'DB / 2*/
  double scale = b + sums2/2.0 + BtDiB/2.0;
  
  /* draw the sample and return it */
  s2 = 1.0/rgamma(shape, 1.0/scale);

  /* check for a problem */
  if(scale <= 0) error("ill-posed regression in DrawS2, scale <= 0");
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

  /* sanity checks */
  assert(lambda2 > 0 && tau2i != NULL);

  /* part of the mu parameter to the inv-gauss distribution */
  l_numer = log(lambda2) + log(s2);

  for(unsigned int j=0; j<m; j++) {
      
    /* the rest of the mu parameter */
    l_mup = 0.5*l_numer - log(fabs(beta[j])); 
    
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
  if(!Compute()) error("ill-posed regression in DrawTau2i");
 }


/*
 * DrawLambda2:
 *
 * Gibbs draw for the lambda2 scalar conditional on the
 * other Bayesian lasso parameters
 */

void Blasso::DrawLambda2(void)
{
  /* do nothing if M = 0 */
  if(M == 0) return;

  /* not checking m > 0 since lambda2 should walk in this case */
  if(lambda2 <= 0) { assert((m == 0 && reg_model == RIDGE) || 
			    reg_model == OLS); return; }
  assert(reg_model != OLS);

  /* see if we're doing lasso or ridge */
  if(reg_model == LASSO) { /* lasso */
   
    /* sanity check */
    if(tau2i == NULL) assert(m == 0);
  
    /* set up gamma distribution parameters */
    double shape = (double) m + r;
    double rate = 0.0;
    for(unsigned int i=0; i<m; i++) {
      if(tau2i[i] == 0) {shape--; continue;}  /* for numerical problems */
      rate += 1.0/tau2i[i];
    }
    rate = rate/2.0 + delta;
  
    /* draw from a gamma distribution */
    lambda2 = rgamma(shape, 1.0/rate);

  } else { /* ridge */

    /* no lambda2 parameter draws for RIDGE when m == 0 */
    if(m == 0) { assert(lambda2 == 0); return; }

    /* sanity check */
    assert(tau2i == NULL && reg_model != OLS);
    
    /* set up Inv-Gamma distribution parameters */
    double BtB = linalg_ddot(m, beta, 1, beta, 1);
    double shape = (double) (m + r)/2;
    double scale = (BtB/s2 + delta)/2;

    /* draw from an Inv-Gamma distribution */
    lambda2 = 1.0/rgamma(shape, 1.0/scale);

    /* lambda2 has changed so need to update beta params */
    if(!Compute() || BtB/s2 <= 0) 
      error("ill-posed regression in DrawLambda2, BtB=%g, s2=%g, m=%d",
	    BtB, s2, m);
  }
}


/*
 * PrintInputs:
 *
 * print the design matrix (X) and responses (Y)
 */

void Blasso::PrintInputs(FILE *outfile) const
{
  /* print the design matrix */
  if(Xorig) {
    myprintf(outfile, "X =\n");
    printMatrix(Xorig, n, M, outfile);
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
  return log_posterior(n, m, resid, beta, s2, tau2i, lambda2,
		       a, b, r, delta, Mmax, mprior);
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
		     const double delta, const unsigned int Mmax, 
		     const double mprior)
{
  /* for summing in the (log) posterior */
  double lpost = 0.0;

  /* calculate the likelihood prod[N(resid | 0, s2)] */
  double sd = sqrt(s2);
  for(unsigned int i=0; i<n; i++) 
    lpost += dnorm(resid[i], 0.0, sd, 1);

  // myprintf(stdout, "lpost +resid = %g\n", lpost);
 
  /* add in the prior for beta */
  if(tau2i) { /* under the lasso */
    for(unsigned int i=0; i<m; i++) 
      if(tau2i[i] > 0)
	lpost += dnorm(beta[i], 0.0, sd*sqrt(1.0/tau2i[i]), 1);
  } else if(lambda2 > 0) { /* under ridge regression */
    for(unsigned int i=0; i<m; i++) 
      lpost += dnorm(beta[i], 0.0, sd*sqrt(lambda2), 1);
  } /* nothing to do under flat/Jeffrey's OLS prior */

  // myprintf(stdout, "lpost +beta = %g\n", lpost);
   
  /* add in the prior for s2 */
  if(a != 0 && b != 0) 
    lpost += dgamma(1.0/s2, a, 1.0/b, 1);
  else lpost += 0.0 - log(s2);  /* Jeffrey's */

  // myprintf(stdout, "lpost +s2 = %g\n", lpost);

  /* add in the prior for tau2 */
  if(tau2i && lambda2 != 0)
    for(unsigned int i=0; i<m; i++)
      lpost += dexp(1.0/tau2i[i], 2.0/lambda2, 1);

  // myprintf(stdout, "lpost +tau2i = %g\n", lpost);

  /* add in the lambda prior */
  if(tau2i) { /* lasso */
    if(lambda2 != 0 && r != 0 && delta != 0) 
      lpost += dgamma(lambda2, r, 1.0/delta, 1); /* is Gamma */
  } else if(lambda2 != 0) { /* ridge */
    if(r != 0 && delta != 0) 
      lpost += dgamma(1.0/lambda2, r, 1.0/delta, 1); /* is Inv-gamma */
    else lpost += 0.0 - log(lambda2); /* Jeffrey's */
  }

  // myprintf(stdout, "lpost +lambda2 (%g) = %g\n", lambda2, lpost);

  /* add in the the model probability */
  lpost += lprior_model(m, Mmax, mprior);

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


/*
 * Method:
 *
 * return an integer encoding of a summary of the
 * typo of regression model employed by this model
 */

int Blasso::Method(void)
{
  if(M == 0) return 1; /* complete */

  if(RJ) { /* reversible jump */
    
    if(reg_model == LASSO) return 2;  /* rjlasso */
    else if(reg_model == RIDGE) return 3;  /* rjridge */
    else return 4;

  } else { /* no RJ */

    if(reg_model == LASSO) return 5; /* lasso */
    if(reg_model == RIDGE) return 6;
    else return 7; /* ols */
  }
}


/*
 * UsesRJ:
 *
 * return the RJ boolean
 */

bool Blasso::UsesRJ(void)
{
  return RJ;
}


/*
 * Verb:
 *
 * return the verbosity argument 
 */

int Blasso::Verb(void)
{
  return verb;
}


/*
 * Thin:
 *
 * Dynamically calculate a number of thinning MCMC rounds
 * by taking RJ and LASSO into account
 */

unsigned int Blasso::Thin(unsigned int thin)
{
  if(RJ || reg_model == LASSO) thin = thin*Mmax;
  else if(reg_model == RIDGE) thin *= 2;
  if(thin == 0) thin++;
  return thin;
}


/*
 * lprior_model:
 *
 * calculate the log prior probability of a regression model
 * with m covariates which is either Binomial(m[Mmax,mprior])
 * or is uniform over 0,...,Mmax
 */

double lprior_model(const unsigned int m, const unsigned int Mmax, 
		    const double mprior)
{
  assert(mprior >= 0 && mprior <= 1);
  if(mprior == 0) return 0;
  else return dbinom((double) m, (double) Mmax, (double) mprior, 1);
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


extern "C"
{
/*
 * lasso_draw_R
 *
 * function currently used for testing the above functions
 * using R input and output
 */

/* global variables, so they can be cleaned up */
double **X = NULL;
double **beta_mat;
double  **tau2i_mat;
Blasso *blasso = NULL;

void blasso_R(int *T, int *thin, int *M, int *n, double *X_in, 
	      double *Y, double *lambda2, double *mu, int *RJ, 
	      int *Mmax, double *beta, int *m, double *s2, 
	      double *tau2i, double *lpost, double *mprior, double *r, 
	      double *delta, double *a, double *b, int *rao_s2, 
	      int *normalize, int *verb)
{
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
  if(tau2i != NULL) { /* for lasso */
    tau2i_mat = (double **)  malloc(sizeof(double*) * (*T));
    tau2i_mat[0] = &(tau2i[*M]);
    for(i=1; i<(*T)-1; i++) tau2i_mat[i] = tau2i_mat[i-1] + (*M);
  } else tau2i_mat = NULL; /* for ridge */

  /* starting and sampling lambda2 if not null */
  double lambda2_start = 0.0;
  double *lambda2_samps = NULL;
  if(lambda2 != NULL) {
    lambda2_start = lambda2[0];
    lambda2_samps = &(lambda2[1]);
  }

  /* create a new Bayesian lasso regression */
  blasso =  new Blasso(*M, *n, X, Y, (bool) *RJ, *Mmax, beta_mat[0], 
		       lambda2_start, s2[0], tau2i, *mprior, *r, *delta, 
		       *a, *b, (bool) *rao_s2, (bool) *normalize, *verb);

  /* part of the constructor which could fail has been moved outside */
  blasso->Init();

  /* Gibbs draws for the parameters */
  blasso->Rounds((*T)-1, *thin, lambda2_samps, &(mu[1]), &(beta_mat[1]), 
		 &(m[1]), &(s2[1]), tau2i_mat, &(lpost[1]));

  delete blasso;
  blasso = NULL;

  /* give the random number generator state back to R */
  PutRNGstate();

  /* clean up */
  free(X); X = NULL;
  free(beta_mat); beta_mat = NULL;
  free(tau2i_mat); tau2i_mat = NULL;
}


/*
 * blasso_cleanup
 *
 * function for freeing memory when blasso is interrupted
 * by R, so that there won't be a (big) memory leak.  It frees
 * the major chunks of memory, but does not guarentee to 
 * free up everything
 */

void blasso_cleanup(void)
{
  /* free blasso model */
  if(blasso) { 
    if(blasso->Verb() >= 1)
      myprintf(stderr, "INTERRUPT: blasso model leaked, is now destroyed\n");
    delete blasso; 
    blasso = NULL; 
  }
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
