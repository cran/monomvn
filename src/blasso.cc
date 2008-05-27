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


extern "C"
{
#include "rhelp.h"
#include "matrix.h"
#include "linalg.h"
#include "Rmath.h"
#include "R.h"
#include "assert.h"
#include "nu.h"
#include "hshoe.h"
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

Blasso::Blasso(const unsigned int M, const unsigned int N, double **Xorig, 
	       Rmiss *R, double *Xnorm, const double Xnorm_scale, 
	       double *Xmean, const unsigned int ldx, double *Y, 
	       const bool RJ, const unsigned int Mmax, double *beta_start, 
	       const double s2_start, const double lambda2_start, 
	       double *mprior, const double r, const double delta, 
	       const double theta, const REG_MODEL reg_model, int *facts, 
	       const unsigned int nf, const bool rao_s2, 
	       const unsigned int verb)
{
  /* sanity checks */
  if(Mmax >= N) 
    assert(RJ || reg_model == LASSO || reg_model == LASSO 
	   || reg_model == RIDGE || (reg_model == FACTOR && nf < N));

  /* copy RJ setting */
  this->RJ = RJ;
  this->reg_model = reg_model;
  this->omega2 = this->tau2i = this->beta = this->rn = this->BtDi = NULL;
  this->lpost = -1e300*1e300;

  /* initialize the parameters */
  this->r = r;
  this->delta = delta;
  this->theta = theta;
  this->nu = 1.0/theta;
  this->icept = (this->theta != 0);

  /* initialize the active set of columns of X */
  pb = NULL; pin = pout = NULL;
  InitIndicators(M, Mmax, beta_start, facts, nf);

  /* copy the Rao-Blackwell option for s2 */
  this->rao_s2 = rao_s2;

  /* initialize the input data */ 
  InitX(N, Xorig, R, Xnorm, Xnorm_scale, Xmean, ldx, true);

  /* copy verbosity argument */
  this->verb = verb;

  /* initialize the mprior value */
  dupv(this->mprior, mprior, 2);
  if(mprior[1] == 0) pi = mprior[0]; /* fixed pi */
  else pi = mprior[0]/(mprior[0] + mprior[1]); /* use mean of a beta distn */

  /* this function will be modified to depend on whether OLS 
     must become lasso */
  InitParams(reg_model, beta_start, s2_start, lambda2_start);

  /* Y initization must come after InitParams */
  InitY(N, Y);

  /* need to help with a stronger s2-prior beta parameter when p>n;
     must happen after Init Y  */
  if((reg_model != OLS && !RJ && M >= n) || theta != 0) {
      a = 3.0/2.0;
      double YtY = linalg_ddot(n, Y, 1, Y, 1);
      b = Igamma_inv(a, 0.95*gammafn(a), 0, 0)*YtY;
  }

  /* only used by one ::Draw function */
  Xbeta_v = new_vector(N);

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
	       double *tau2i, const bool hs, double *omega2, const double nu, 
	       double *mprior, const double r,  const double delta, 
	       const double a, const double b, const double theta, 
	       const bool rao_s2, const bool normalize, const unsigned int verb)
{
  /* copy RJ setting */
  this->RJ = RJ;
  this->reg_model = OLS;  /* start as != FACTOR and update in InitParams */
  this->omega2 = this->tau2i = this->beta = this->rn = this->BtDi = NULL;
  this->lpost = -1e300*1e300;

  /* initialize the parameters */
  this->r = r;
  this->delta = delta;
  this->theta = theta;
  this->icept = (this->theta != 0.0);

  /* initialize the active set of columns of X */
  pb = NULL; pin = pout = NULL;
  InitIndicators(M, Mmax, beta, NULL, 0);

  /* copy the Rao-Blackwell option for s2 */
  this->rao_s2 = rao_s2;

  /* initialize the input data */ 
  InitX(n, X, normalize);

  /* copy verbosity argument */
  this->verb = verb;

  /* initialize the mprior value */
  dupv(this->mprior, mprior, 2);
  if(mprior[1] == 0) pi = mprior[0];
  else pi = mprior[0]/(mprior[0] + mprior[1]); /* use mean of a beta distn */

  /* this function will be modified to depend on whether OLS 
     must become lasso */
  InitParams(beta, lambda2, s2, tau2i, hs, omega2, nu); 

  /* Y initization must come after InitParams */
  InitY(N, Y);

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
  if(!Compute(false)) error("ill-posed regression in Init");
}


/*
 * InitIndicators:
 *
 * set the total number of colums, M, and then set the
 * initial number of non-zero columns, m, and meanwhile
 * choose the starting configuration of the model by
 * choosing the columns of the design matrix Xp based on
 * (either) the non-zero components of beta, or the
 * designated factors, or a nearly-saturate model based
 * on Mmax
 */

void Blasso::InitIndicators(const unsigned int M, const unsigned int Mmax, 
			    double *beta, int *facts, const unsigned int nf)
{
  /* copy the dimension parameters*/
  this->M = M;
  this->Mmax = Mmax;

  /* sanity checks */
  assert(Mmax <= M);
  if(!RJ) assert(Mmax == M);
  if(reg_model == FACTOR && M >= 1) assert(facts);

  /* allocate and initialize the pb vector; this sets m */
  InitPB(beta, facts, nf);

  /* allocate the column indicators and fill them */
  /* start with the in-indicators; m set in InitPB */
  pin = new_ivector(m);
  unsigned int j=0;
  for(unsigned int i=0; i<M; i++) if(pb[i]) pin[j++] = i;
  assert(j == m);

  /* handing the out-indicators may depend on the factors available */
  if(reg_model == FACTOR) {

    /* allocate pout vector and sanity check */
    assert(nf >= m);
    pout = new_ivector(nf-m);
    
    /* loop over each factor in the available columns */
    unsigned int k=0;
    for(unsigned int i=0; i<nf; i++)
      if(facts[i] < (int) M && pb[facts[i]] == false) pout[k++] = facts[i];
    assert(k == this->nf-m);

  } else { /* simply allocate pout and fill with all false pb */
    pout = new_ivector(M-m);
    unsigned int k = 0;
    for(unsigned int i=0; i<M; i++) if(!pb[i]) pout[k++] = i;
    assert(k == M-m);
  }
}


/*
 * InitPB:
 *
 * auxilliary function used by InitIndicators to calculate the
 * (default) initial setting of the boolean indicator vector
 * pb, describing which columns of Xorig are to be used in the
 * initial model
 */

void Blasso::InitPB(double *beta, int *facts, const unsigned int nf)
{
  /* sanity check */
  assert(pb == NULL);

  /* find out which betas are non-zero, thus setting m */
  pb = (bool*) malloc(sizeof(bool) * M);

  /* start by filling in pb with factors on */
  unsigned int j = 0;
  for(unsigned int i=0; i<M; i++) pb[i] = false;
  for(unsigned int i=0; i<nf; i++) /* put true for each factor */
    if(facts[i] < (int) M) { pb[facts[i]] = true; j++; }

  /* the resulting j should be the number of factors in this regression */
  assert(j <= nf);
  this->nf = j;

  /* then force that the specified Mmax is appropriate for this case */
  if(reg_model == FACTOR && this->Mmax > j) this->Mmax = j;

  /* check if model initialization is specified by 
     an initial beta vector */
  if(beta != NULL) {

    /* fill in the indicators to true where beta[i] != NULL */
    m = 0;
    for(unsigned int i=0; i<M; i++) {
      if(beta[i] != 0) { 
	if(facts && pb[i] != true) 
	  warning("starting beta[%d] != 0 and col %d is not a factor");
	else { pb[i] = true; m++; }
      } else pb[i] = false;
      assert(m <= this->Mmax);
    }

    /* see if we are starting in a non-saturated model when RJ
       is false, and warn if so */
    if(!RJ && m < M) 
      warning("RJ=FALSE, but not in saturated model (m=%d, M=%d), try RJ=\"p\"",
	      m, M);

  } else { /* otherwise default depends on RJ */

    /* if using RJ, then start nearly saturated */
    if(RJ) m = (unsigned int) (0.75* this->Mmax);
    else m = this->Mmax;  /* otherwise start saturated */

    /* fill in the corresponding booleans */
    if(reg_model == FACTOR) { /* unset j-Mmax booleans */
      for(j=this->nf; j> this->Mmax; j--) {
	assert(pb[facts[j]] == true);
	pb[facts[j]] = false;
      }
    } else { /* general case */
      for(unsigned int i=0; i<m; i++) pb[i] = true;
      for(unsigned int i=m; i<M; i++) pb[i] = false;
    }
  }
}


/* 
 * InitX:
 *
 * initialize the input data (X and Y) by allocating new
 * memory and calculating normalized stuff
 */

void Blasso::InitX(const unsigned int N, double **Xorig,
		    const bool normalize)
{
  this->ldx = M;
  this->copies = true;
  this->n = N;
  this->N = N;

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
  Xp = new_p_submatrix(pin, X, n, m, icept);
  if(icept) for(unsigned int i=0; i<n; i++) Xp[i][0] = 1.0;
  delete_matrix(X);
  /* DiXp is set in ::InitY() after ::InitParams() */

  /* set R and n2 equal to default NULL values */
  R = NULL;

  /* wait 'till after params have been set to initialize Y and
     DiXp, which depends on omega */
  DiXp = NULL;
}


/* 
 * InitX:
 *
 * initialize the input data (X and Y) from pointers allocated
 * outside this module (likely in Bmonomvn::)
 */

void Blasso::InitX(const unsigned int N, double **Xorig, Rmiss *R, 
		   double *Xnorm, const double Xnorm_scale, 
		   double *Xmean, const unsigned int ldx, const bool normalize)
{
  this->copies = false;

  /* number of rows in design matrix depends on which responses require DA */
  this->N = N;
  this->R = R;
  if(R) this->n = this->N - R->n2[M];
  else this->n = this->N;

  /* copy the POINTER to the input matrices */
  /* NOTE THAT Xorig AND X WILL HAVE LARGER LEADING DIMENSIONS */
  this->Xorig = Xorig;
  this->Xmean = Xmean;
  this->normalize = normalize;
  this->Xnorm = Xnorm;
  this->Xnorm_scale = Xnorm_scale;
  this->ldx = ldx;

  /* extract the active columns of X; and acrive rows if(Rmiss) */
  Xp = new_matrix(n, m+icept);
  if(icept) for(unsigned int i=0; i<n; i++) Xp[i][0] = 1.0;
  unsigned int k, ell;
  k = ell = 0;
  unsigned int *R2 = NULL;
  if(R) R2 = R->R2[M];

  /* for each fow of Xorig */
  for(unsigned int i=0; i<N; i++) {

    /* skipping Rt[M][] == 2 */
    if(R2 && ell < R->n2[M] && R2[ell] == i) { ell++; continue; } 
    
    /* copying from Xorig to Xp with centering and normalization */
    for(unsigned int j=0; j<m; j++) {
      Xp[k][j+icept] = Xorig[i][pin[j]] - Xmean[pin[j]];
      if(normalize) Xp[k][j+icept] /= Xnorm_scale * Xnorm[pin[j]];
    }
    k++;
  }
  assert(k == n);

  /* wait 'till after params have been set to initialize Y and
     DiXp, which depends on omega */
  DiXp = NULL;
}


/* 
 * InitY:
 *
 * handle the rest of the XY initialization that has to do with
 * Y -- must happen after all X initialization is done (InitX)
 * and parameters are initialized (InitParas)
 */

void Blasso::InitY(const unsigned int N, double *Y)
{	   
  /* sanity check */
  if(!R) assert(this->n == N);
  else assert(this->n == (N - R->n2[M]));

  /* center Y, and account for Rmiss if necessary */
  this->Y = new_vector(n);
  unsigned int k, ell;
  k = ell = 0;
  unsigned int* R2 = NULL;
  if(R) R2 = R->R2[M];

  /* for each entry of Y */
  Ymean = 0.0;
  for(unsigned int i=0; i<N; i++) {

    /* skip this row if Rt[M] == 2 */
    if(R2 && ell < R->n2[M] && R2[ell] == i) { ell++; continue; }

    /* copy Y to this->Y */
    this->Y[k] = Y[i];
    Ymean += this->Y[k];
    k++;
  }
  assert(k == n);
  Ymean /= n;
  
  /* do not center Y if there is an intercept in the model */
  if(!icept) centerv(this->Y, n, Ymean); 
  else beta[0] = Ymean;

  /* initialize the residual vector */
  resid = new_dup_vector(this->Y, n);
  if(m+icept > 0) linalg_dgemv(CblasTrans,m+icept,n,-1.0,Xp,m+icept,beta,1,1.0,resid,1);

  /* possibly create the DiXp matrix, initialized to Xp 
     -- filled in UpdateXY below */
  if(theta != 0) {
    assert(omega2 != NULL);
    DiXp = new_zero_matrix(n, m+icept);
  } else DiXp = NULL;

  /* for calculating t(X) %*% Y -- filled in UpdateXY */
  XtY = new_zero_vector(m+icept);

  /* actually calculate the quantities allocated immediately above */
  UpdateXY();
}


/* 
 * UpdateXY:
 *
 * update functions of Xp and Y which may depend on omega2,
 * or are simply being initialized by InitY
 */

void Blasso::UpdateXY(void)
{
  /* update DiXpt if omega2 != NULL */
  if(this->DiXp) {
    assert(omega2);
    for(unsigned int i=0; i<n; i++) 
      for(unsigned int j=0; j<m+icept; j++) 
	this->DiXp[i][j] = Xp[i][j]/omega2[i];
  }
  
  /* Use DiXp below if omega2 != NULL */
  double **DiXp = this->DiXp;
  if(DiXp == NULL) DiXp = Xp;
  else assert(omega2 != NULL);

  /* calculate t(X) %*% Y */
  if(XtY) linalg_dgemv(CblasNoTrans,m+icept,n,1.0,DiXp,m+icept,this->Y,1,0.0,XtY,1);

  /* calculate YtY possibly using imega */
  if(omega2 != NULL) {
    YtY = 0.0;
    for(unsigned int i=0; i<n; i++) YtY += Y[i]*Y[i]/omega2[i];
  } else YtY = linalg_ddot(n, this->Y, 1, this->Y, 1);
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
  breg = new_BayesReg(m+icept, n, Xp, DiXp);

  /* allocate the other miscellaneous vectors needed for
     doing the regressions */
  assert(BtDi == NULL && rn == NULL);
  if(!rao_s2) BtDi = new_vector(m+icept);
  else BtDi = NULL;
  rn = new_vector(M+icept);
}


/*
 * new_BayesReg:
 *
 * allocate a new regression utility structure
 */

BayesReg* new_BayesReg(const unsigned int m, const unsigned int n, 
		       double **Xp, double **DiXp)
{
  /* allocate the structure */
  BayesReg *breg = (BayesReg*) malloc(sizeof(struct bayesreg));
  breg->m = m;

  /* allocate A and XtX-diag */
  breg->A = new_zero_matrix(m, m);
  breg->XtX_diag = new_vector(m);

  /* fill A and XtX-diag */
  init_BayesReg(breg, m, n, Xp, DiXp);
  
  /* allocate the rest and return */
  alloc_rest_BayesReg(breg);
  return(breg);
}



/*
 * init_BayesReg:
 *
 * fill an already allocated a new regression utility structure
 * with A and XtX-diag
 */

BayesReg* init_BayesReg(BayesReg *breg, const unsigned int m, 
			const unsigned int n, double **Xp, double **DiXp)
{
  /* sanity checks */
  assert(breg->m == m);
  if(m != 0) assert(breg->A);

  /* if not using omega2 */
  if(DiXp == NULL) DiXp = Xp;
  
  /* fill A with t(X) %*% Di %*% X -- i.e. XtDxsX */
  if(breg->A)
    linalg_dgemm(CblasNoTrans,CblasTrans,m,m,n,1.0,
		 Xp,m,DiXp,m,0.0,breg->A,m);
  
  /* save the diagonal of XtX in a separate vector */
  if(m != 0) assert(breg->XtX_diag);
  for(unsigned int i=0; i<m; i++) breg->XtX_diag[i] = breg->A[i][i];

  /* return for continued use or finishing allocation */
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
			 BayesReg *old, double *xnew, double **Xp, 
			 double *omega2)
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
  if(omega2) {
    breg->A[m][m] = 0.0;
    for(unsigned int i=0; i<n; i++) 
      breg->A[m][m] += xnew[i]*xnew[i]/omega2[i];
  } else breg->A[m][m] = linalg_ddot(n, xnew, 1, xnew, 1);

  /* get the new row of XtX_new[col,] = X[,col] %*% X[,1:m] */
  for(unsigned int j=0; j<m; j++) {
    breg->A[m][j] = 0;
    unsigned int i;
    if(omega2) for(i=0; i<n; i++) breg->A[m][j] += Xp[i][j]*xnew[i]/omega2[i];
    else for(i=0; i<n; i++) breg->A[m][j] += Xp[i][j]*xnew[i];
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
 * of model -- called by ::Bmonomvn
 */

void Blasso::InitParams(REG_MODEL reg_model, double *beta, double s2,
			double lambda2)
{
  /* sanity check */
  assert(this->tau2i == NULL && this->beta == NULL && this->omega2 == NULL);
  assert(reg_model == this->reg_model);

  assert(reg_model != HORSESHOE); // NOT HANDLED YET 

  /* set the LASSO/HORSESHOE & RIDGE lambda2 and tau2i initial values */
  if(reg_model != OLS) {

    /* assign starting lambda2 value */
    this->lambda2 = lambda2;
    if(m > 0 && lambda2 <= 0  && reg_model == LASSO) {
      warning("starting lambda2 (%g) <= 0 is invalid (m=%d, M=%d)", 
	      lambda2, m, M);
      this->lambda2 = 1.0;
    } else this->lambda2 = lambda2;

    /* tau2i is only used by LASSO, not RIDGE */
    if(reg_model == LASSO) {
      tau2i = ones(m+icept, 1.0);
      if(icept) tau2i[0] = 0.0;
    } else { /* for RIDGE */
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
  this->beta = new_zero_vector(m+icept);

  /* initialize the initial beta vector */
  if(beta) {
    /* norm the beta samples, like Efron and Hastie */
    if(normalize && this->m > 0) {
      scalev2(beta, M, Xnorm);
      scalev(beta, M, Xnorm_scale);
    }
    
    /* copy in the beta vector */
    copy_sub_vector((this->beta)+icept, pin, beta, m);
  } else {
    /* initialize beta so that it is non-zero */
    for(unsigned int i=0; i<m; i++) this->beta[i+icept] = 1.0;
  }

  /* initialize regression coefficients */
  this->s2 = s2;

  /* default setting when not big-p-small-n */
  a = b = 0;

  /* intialize omega2 */
  if(theta != 0) omega2 = ones(n, theta);
  nu = 1.0/theta;
}



/*
 * InitParams:
 *
 * set specific values of the starting values, and the 
 * dynamically determine the type of model
 */

void Blasso::InitParams(double *beta, const double lambda2, 
			const double s2, double *tau2i, const bool hs, 
			double *omega2, double nu)
{
  this->lambda2 = lambda2;
  this->s2 = s2;

  /* copy in the tau2i vector */
  assert(this->tau2i == NULL);
  if(tau2i != NULL) {
    this->tau2i = new_vector(m+icept);
    if(icept) this->tau2i[0] = 0;
    copy_sub_vector((this->tau2i)+icept, pin, tau2i, m);
  }

  /* allocate beta */
  this->beta = new_vector(m+icept);
  if(icept) this->beta[0] = Ymean;

  /* norm the beta samples, like Efron and Hastie */
  if(normalize && this->m > 0) {
    scalev2(beta, M, Xnorm);
    scalev(beta, M, Xnorm_scale);
  }

  /* copy in the beta vector */
  copy_sub_vector((this->beta)+icept, pin, beta, m);

  /* determine the resulting regression model */
  if(lambda2 == 0) {
    reg_model = OLS;
    assert(tau2i == NULL);
  } else if(tau2i == NULL) reg_model = RIDGE;
  else {
    if(m > 0) assert(sumv(this->tau2i, m+icept) != 0);
    if(hs) reg_model = HORSESHOE;
    else reg_model = LASSO;
  }

  /* set lambda2 to zero if m == 0  && RIDGE */
  if(M == 0 || (reg_model == RIDGE && m == 0)) this->lambda2 = 0.0;

  /* intialize omega2 */
  assert(this->omega2 == NULL);
  if(theta != 0) this->omega2 = new_dup_vector(omega2, n);
  this->nu = nu;
}


/*
 * ~Blasso:
 *
 * the usual destructor function
 */

Blasso::~Blasso(void)
{
  /* clean up */
  if(omega2) free(omega2);
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
  if(DiXp) delete_matrix(DiXp);
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
 * GetParams:
 * 
 * get the current values of the parameters to the pointers
 * to memory provided -- assumes beta & tau2i are m-vectors
 */

void Blasso::GetParams(double *mu, double *beta, int *m, double *s2, 
		       double *tau2i, double *omega2, double *nu, 
		       double *lambda2, double *pi) const
{
  /* copy back the intercept */
  if(icept) *mu = this->beta[0];
  else *mu = rnorm(Ymean, sqrt(this->s2/n));

  /* copy back the regression coefficients and non-zero count */
  *m = this->m;
  zerov(beta, M);
  if(this->m > 0) copy_p_vector(beta, pin, (this->beta)+icept, this->m);

  /* copy back the beta error structure and laplace prior */
  *s2 = this->s2;
  if(tau2i && (reg_model == LASSO || reg_model == HORSESHOE)) {
    for(unsigned int i=0; i<M; i++) tau2i[i] = -1.0;
    if(this->m > 0 && this->tau2i)
      copy_p_vector(tau2i, pin, (this->tau2i)+icept, this->m);
  }
  if(lambda2) *lambda2 = this->lambda2;

  /* copy back the omega2 latent variables for the Student-t prior,
     and the degrees of freedom nu */
  if(omega2 && this->omega2) dupv(omega2, this->omega2, n);
  if(nu) *nu = this->nu;
  
  /* copy back pi */
  if(pi) *pi = this->pi;
}


/*
 * PrintParams:
 * print the current values of the parameters to the
 * specified file
 */

void Blasso::PrintParams(FILE *outfile) const
{
  myprintf(outfile, "m=%d, lambda2=%g, s2=%g, icept=%d\n", m, lambda2, s2, icept);
  myprintf(outfile, "beta = ");
  printVector(beta, m+icept, outfile, HUMAN);
  if(tau2i) {
    myprintf(outfile, "tau2i = ");
    printVector(tau2i, m+icept, outfile, HUMAN);
  }
  if(omega2) {
    myprintf(outfile, "omega2 = ");
    printVector(omega2, n, outfile, HUMAN);
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
		    double *mu, double **beta, int *m, double *s2, 
		    double **tau2i, double *lambda2, double **omega2, 
		    double *nu, double *pi, double *lpost, 
		    double *llik, double *llik_norm)
{
  /* sanity check */
  assert(breg);

  /* for helping with periodic interrupts */
  time_t itime = time(NULL);

  /* assume that the initial values reside in position 0 */
  /* do T-1 MCMC rounds */
  for(unsigned int t=0; t<T; t++) {
    
    /* do thin number of MCMC draws */
    Draw(thin, false);

    /* if LASSO then get t-th tau2i */
    double *tau2i_samp = NULL;
    if(tau2i) { 
      assert(reg_model == LASSO || reg_model == HORSESHOE); 
      tau2i_samp = tau2i[t]; 
    }

    /* if doing Student-t scale mixcures then get t-th omega2 */
    double *omega2_samp = NULL;
    double *nu_samp = NULL;
    if(omega2) { 
      assert(nu && theta != 0); 
      omega2_samp = omega2[t]; 
      nu_samp = &(nu[t]);
    }

    /* if LASSO, horseshoe or ridge */
    double *lambda2_samp = NULL;
    if(lambda2) { 
      assert(m == 0 || reg_model != OLS); 
      lambda2_samp = &(lambda2[t]); 
    } 

    /* if pi not fixed */
    double *pi_samp = NULL;
    if(mprior[1] != 0) pi_samp = &(pi[t]);

    /* copy the sampled parameters */
    GetParams(&(mu[t]), beta[t], &(m[t]), &(s2[t]), tau2i_samp, 
	      omega2_samp, nu_samp, lambda2_samp, pi_samp);
    
    /* get the log posterior */
    lpost[t] = this->lpost;
    llik[t] = this->llik;
    if(omega2) llik_norm[t] = this->llik_norm;
    
    /* print progress meter */
    if(verb && t > 0 && ((t+1) % 100 == 0))
      myprintf(stdout, "t=%d, m=%d\n", t+1, this->m);

    /* check R for interrupts and flush console every second */
    itime = my_r_process_events(itime);
  }

  /* (un)-norm the beta samples, like Efron and Hastie */
  if(normalize) {
    norm_columns(beta, Xnorm, T, M);
    scalev(beta[0], T*M, 1.0/Xnorm_scale);
  }

  /* calculate mu samples */
  assert(R == NULL);

  /* adjustment to mu needed since Xp used centered columns of Xorig */
  /* Xbeta = X %*% t(beta), in col-major representation */
  double **Xbeta = new_zero_matrix(T,n);
  linalg_dgemm(CblasTrans,CblasNoTrans,n,T,M,1.0,Xorig,ldx,beta,M,0.0,Xbeta,n);
  /* mu = apply(Xbeta, 2, mean), with Xbeta in col-major representation */
  double* mu_resid = new_vector(T);
  wmean_of_rows(mu_resid, Xbeta, T, n, NULL);
  /* adjustmment performed below */
  
  /* mu = rnorm(rep(1,Ymean), sqrt(s2/n)) - apply(Xbeta, 2, mean) */
  for(unsigned t=0; t<T; t++) {
    if(!icept) lpost[t] += dnorm(mu[t], Ymean, sqrt(s2[t]/n), 1);
    mu[t] = mu[t] - mu_resid[t]; /*adjustment performed here */
  }
    
  /* clean up */
  delete_matrix(Xbeta);
  free(mu_resid);
}


/*
 * Draw:
 *
 * Gibbs draws for each of the bayesian lasso parameters
 * in turn: beta, s2, tau2i, and lambda2;  the thin paramteters
 * causes thin-1 (number of) draws to be burned first
 */

void Blasso::Draw(const unsigned int thin, const bool fixnu)
{
  /* sanity check */
  assert(thin > 0);

  for(unsigned int t=0; t<thin; t++) {

    /* draw from the model prior parameter */
    if(RJ) DrawPi();

    /* draw the latent Student-t variables */
    if(omega2 && !isinf(nu)) DrawOmega2();

    /* draw latent lasso variables, and update Bmu and Vb, etc. */
    if(reg_model == LASSO || reg_model == HORSESHOE) DrawTau2i();
    else assert(tau2i == NULL);

    /* recompute the BayesReg module since omega and/or tau2i have changed */
    if(omega2 && tau2i && !Compute(true))
      error("ill-posed regression in DrawTau2i or DrawOmega2");
    else if(omega2 && !Compute(true))
      error("ill-posed regression in DrawOmega2");
    else if(tau2i && !Compute(false))
      error("ill-posed regression in DrawTau2i");
    
    /* draw nu based on the omega2s */
    if(!isinf(nu) && omega2 && !fixnu) DrawNu();

    /* only depends on tau2i for LASSO & HORSESHOE, and beta for RIDGE */
    if(reg_model != OLS) DrawLambda2();
    else { /* is OLS */
      assert(lambda2 == 0 && tau2i == NULL);
      if(m+icept > 0) refresh_Vb(breg, s2);
    }

    /* depends on pre-calculated bmu, Vb, etc, which depends
       on tau2i -- breg->Vb then becomes decomposed */
    DrawBeta();

    /* resid = X*beta - Y */
    dupv(resid, Y, n);
    if(m+icept > 0) linalg_dgemv(CblasTrans,m+icept,n,-1.0,Xp,m+icept,
				 beta,1,1.0,resid,1);

    /* choose the type of s2 GS update */
    if(rao_s2) {
      DrawS2Margin();  /* depends on bmu and Vb but not beta */
    } else DrawS2();  /* depends on beta */

    /* propose to add or remove a column from the model */
    if(RJ) {
      /* first tally the log posterior value of this sample */
      logPosterior();
      RJmove();
    }
  }

  /* calculate the log posterior if it hasn't already been done */
  if(!RJ) logPosterior();
}


/*
 * Draw:
 *
 * Gibbs draws for each of the Bayesian lasso parameters
 * in turn: beta, s2, tau2i, and lambda2; the thin paramteters
 * causes thin-1 (number of) draws to be burned first, and
 * copy them out to the pointers/memory passed in -- thin has been
 * adapted to be thin*Mmax; this has probably been called by 
 * Bmonomvn::Rounds 
 */

void Blasso::Draw(const double thin, const bool usenu, 
		  double *mu, double *beta, int *m, double *s2, 
		  double *tau2i, double *lambda2, double *omega2, 
		  double *nu, double *pi, double *lpost,
		  double *llik, double *llik_norm)
{
  /* sanity check */
  assert(breg && Xbeta_v);

  /* get DA samples from Xorig */
  DataAugment();

  /* do thin number of MCMC draws */
  if(usenu) this->nu = *nu;
  Draw(Thin(thin), usenu);
  if(usenu) assert(this->nu == *nu);

  /* copy the sampled parameters */
  GetParams(mu, beta, m, s2, tau2i, omega2, nu, lambda2, pi);

  /* (un)-norm the beta samples, like Efron and Hastie */
  if(normalize && this->m > 0) {
    normv(beta, M, Xnorm);
    scalev(beta, M, 1.0/Xnorm_scale);
  }

  /* adjustment to mu needed since Xp used centered columns of Xorig */
  double mu_resid = 0;
  if(this->m > 0) {
    /* Xbeta = X %*% beta, in col-major representation */
    linalg_dgemv(CblasTrans,M,N,1.0,Xorig,ldx,beta,1,0.0,Xbeta_v,1);
    
    /* mu_resid = mean(Xbeta) skipping places where Rt[M] == 2 */
    if(R && R->R2[M]) 
      for(unsigned int i=0; i<R->n2[M]; i++) Xbeta_v[R->R2[M][i]] = 0;
    mu_resid = meanv(Xbeta_v, N);

    /* adjust by the number of places skipped above */
    if(R && R->R2[M]) mu_resid *= ((double)N)/(N - R->n2[M]);
  }
  /* adjustment performed below */

  /* calculate the log posterior */
  *lpost = this->lpost;
  if(thin > 0 && !icept) *lpost += dnorm(*mu, Ymean, sqrt((*s2)/n), 1);
  
  /* mu = mu - mean(Xbeta) */
  *mu = *mu - mu_resid; /* adjustment performed here */

  /* return the log likelihood */
  *llik = this->llik;
  *llik_norm = this->llik_norm;
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
  unsigned int Mavail = M - m;
  if(reg_model == FACTOR) Mavail = nf - m;
  int iout = (int) (Mavail * unif_rand());
  int col = pout[iout];
  qratio *= ((double)Mavail)/(m+1);
  double *xnew = NewCol(col);

  /* randomly propose a new tau2i component or lambda depending on reg_model */
  double lpq_ratio;
  double prop = ProposeTau2i(&lpq_ratio);

  /* add a new component to XtY */
  XtY = (double*) realloc(XtY, sizeof(double)*(m+icept+1));
  if(omega2 != NULL) {
    XtY[m+icept] = 0.0;
    for(unsigned int i=0; i<n; i++) XtY[m+icept] += xnew[i]*Y[i]/omega2[i];
  } else XtY[m+icept] = linalg_ddot(n, xnew, 1, Y, 1);

  /* allocate new regression stuff */
  /* diagonal of A is taken care of inside of compute_BayesReg() */
  BayesReg *breg_new = plus1_BayesReg(m+icept, n, breg, xnew, Xp, omega2);

  /* compute the new regression quantities */
  assert(compute_BayesReg(m+icept+1, XtY, tau2i, lambda2, s2, breg_new));

  /* calculate the acceptance probability breg -> breg_new */
  double lalpha = rj_betas_lratio(breg, breg_new, s2, prop);

  /* add in the forwards and prior probabilities */
  lalpha +=  lpq_ratio;

  /* add in the (log) prior model probabilities */
  lalpha += lprior_model(m+1, Mmax, pi) - lprior_model(m, Mmax, pi);

  /* MH accept or reject */
  if(unif_rand() < exp(lalpha)*qratio) { /* accept */

    /* copy the new regression utility */
    delete_BayesReg(breg); breg = breg_new;
    
    /* draw the new beta vector */
    beta = (double*) realloc(beta, sizeof(double)*(m+icept+1));
    draw_beta(m+icept+1, beta, breg, s2, rn);

    /* calculate new residual vector */
    dupv(resid, Y, n);
    if(m+icept > 0) linalg_dgemv(CblasTrans,m+icept,n,-1.0,Xp,m+icept,beta,1,1.0,resid,1);
    linalg_daxpy(n, 0.0 - beta[m+icept], xnew, 1, resid, 1);

    /* other copies */
    if(BtDi) BtDi = (double*) realloc(BtDi, sizeof(double) * (m+icept+1));

    /* add another column to the design matrix */
    Xp = new_bigger_matrix(Xp, n, m+icept, n, m+icept+1);
    dup_col(Xp, m+icept, xnew, n);
    if(omega2) {
      assert(DiXp);
      DiXp = new_bigger_matrix(DiXp, n, m+icept, n, m+icept+1);
      for(unsigned int i=0; i<n; i++) DiXp[i][m+icept] = xnew[i]/omega2[i];
    }
    add_col(iout, col);

    /* calculate the new log_posterior */
    logPosterior();

  } else { /* reject */
    
    /* realloc vectors */
    if(reg_model == LASSO || reg_model == HORSESHOE) 
      tau2i = (double*) realloc(tau2i, sizeof(double)*(m+icept));
    else if(reg_model == RIDGE && m == 0) lambda2 = 0;
    XtY = (double*) realloc(XtY, sizeof(double)*(m+icept));
    
    /* free new regression utility */
    delete_BayesReg(breg_new);
  }

  /* clean up */
  free(xnew);
}


/*
 * ProposeTau2i:
 *
 * randomly propose a new tau2i-component for LASSO/HORSESHOE, or
 * new lambda2 component for RIDGE when m == 0;  this function
 * would be called exclusively by RJup().  For LASSO/HORSESHOE, 
 * the tau2i vector is increased in length by one.  Also calculates 
 * the (log) ratio of prior to proposal probabilities
 */

double Blasso::ProposeTau2i(double *lpq_ratio)
{
  double prop = 1.0;          /* new value proposed */
  *lpq_ratio = 0.0;           /* log ratio of prior to proposal probabilities */

  /* switch over model choices */
  if(reg_model == LASSO || reg_model == HORSESHOE) { 
    /* propose new m-th component of tau2i */
    tau2i = (double*) realloc(tau2i, sizeof(double)*(m+icept+1));  /* grow tau2i */

    /* sample tau2 from the prior: under horseshoe or lasso  */
    if(reg_model == HORSESHOE) prop = LambdaCPS_prior_draw(lambda2); 
    else  prop = rexp(2.0/lambda2);     

    /* assign to the new last entry of tau2i */    
    tau2i[m+icept] = 1.0 / prop;        
    /* then prior and proposal probabilites cancel */

  } else if(reg_model == RIDGE && m == 0) { /* randomly propose a new lambda */   
    assert(lambda2 == 0);                     /* sanity check */
    if(r != 0 && delta != 0){   /* then sample from the prior */
      prop = 1.0/rgamma(r, 1.0/delta);        /* is Inv-gamma */
      /* then prior to proposal ratios cancel */
    } else {
      prop = rexp(1);  /* otherwise sample from an exponential */
      *lpq_ratio =  0.0 - log(prop) - dexp(prop, 1, 1); 
      /* prior to proposal ratio does not cancel */
    }
    lambda2 = prop;
  } else if(reg_model == RIDGE) prop = lambda2;

  /* return the proposed value */
  return prop;
}


/* 
 * UnproposeTau2i:
 *
 * select corresponding tau2i-component for removal from the vector;
 * remove that component and store the corresponding (reverse) 
 * "proposal", thereby undoing ProposTau2i; this function would be
 * exclusively called form RJdown().  The calculated (log) ratio of 
 * proposal to prior probability is returned 
 */

double Blasso::UnproposeTau2i(double *lqp_ratio, unsigned int iin)
{

  double prop = 1.0;        /* reverse of new value proposed */
  *lqp_ratio = 0.0;         /* proposal to prior ratio */

  /* switch over model choices */
  if(reg_model == LASSO || reg_model == HORSESHOE) {  
    /* unpropose the iin-th component of tau2i */
    prop = 1.0/tau2i[iin+icept];          /* remove from the iin-th position */
    tau2i[iin+icept] = tau2i[m+icept-1];
    tau2i = (double*) realloc(tau2i, sizeof(double)*(m+icept-1));
    /* then the proposal and prior probabilities cancel */
  } else if(reg_model == RIDGE && m == 1) { 
    prop = lambda2; lambda2 = 0.0;
    if(r == 0 || delta == 0) *lqp_ratio = dexp(prop,1,1) + log(lambda2);
    /* otherwise the proposal ratios cancel */
  } else if(reg_model == RIDGE) prop = lambda2;
  
  /* return the un-proposed value */
  return prop;
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
  double **Xp_new, **DiXp_new;
  Xp_new = new_dup_matrix(Xp, n, m+icept-1);
  if(DiXp) DiXp_new = new_dup_matrix(DiXp, n, m+icept-1);
  else DiXp_new = NULL;
  if(iin != ((int)m)-1) {
    for(unsigned int i=0; i<n; i++) Xp_new[i][iin+icept] = Xp[i][m+icept-1];
    if(DiXp_new) for(unsigned int i=0; i<n; i++) DiXp_new[i][iin+icept] = DiXp[i][m+icept-1];
  }
  
  /* un-propose the iin-th element of tau2i */
  double lqp_ratio;
  double prop = UnproposeTau2i(&lqp_ratio, iin);
  
  /* remove component of XtY */
  double xty = XtY[iin+icept];
  XtY[iin+icept] = XtY[m+icept-1];
  XtY = (double*) realloc(XtY, sizeof(double)*(m+icept-1));

  /* allocate new regression stuff */
  BayesReg *breg_new = new_BayesReg(m+icept-1, n, Xp_new, DiXp_new);

  /* compute the new regression quantities */
  bool success = compute_BayesReg(m+icept-1, XtY, tau2i, lambda2, s2, breg_new);
  assert(success);

  /* calculate the acceptance probability breg -> breg_new */
  double lalpha = rj_betas_lratio(breg, breg_new, s2, prop);

  /* add in the backwards and prior probabilities */
  lalpha += lqp_ratio;

  /* add in the (log) prior model probabilities */
  lalpha += lprior_model(m-1, Mmax, pi) - lprior_model(m, Mmax, pi);

  /* MH accept or reject */
  if(unif_rand() < exp(lalpha)*qratio) { /* accept */

    /* copy the new regression utility */
    delete_BayesReg(breg); breg = breg_new;

    /* draw the new beta vector */
    beta = (double*) realloc(beta, sizeof(double)*(m+icept-1));
    draw_beta(m+icept-1, beta, breg, s2, rn);
    
    /* calculate new residual vector */
    dupv(resid, Y, n);
    if(m+icept-1 > 0) linalg_dgemv(CblasTrans,m+icept-1,n,-1.0,Xp_new,m+icept-1,
			     beta,1,1.0,resid,1);

    /* other */
    if(BtDi) BtDi = (double*) realloc(BtDi, sizeof(double) * (m+icept-1));

    /* permanently remove the column from the design matrix */
    delete_matrix(Xp); Xp = Xp_new;
    delete_matrix(DiXp); DiXp = DiXp_new;
    remove_col(iin, col);

    /* calculate the new log_posterior */
    logPosterior();

  } else { /* reject */

    /* realloc vectors */
    if(reg_model == LASSO || reg_model == HORSESHOE) { 
      tau2i = (double*) realloc(tau2i, sizeof(double)*(m+icept));
      tau2i[m+icept-1] = tau2i[iin+icept]; tau2i[iin+icept] = 1.0/prop;
    } else if(reg_model == RIDGE && m == 1) lambda2 = prop;
    XtY = (double*) realloc(XtY, sizeof(double)*(m+icept));
    XtY[m+icept-1] = XtY[iin+icept]; XtY[iin+icept] = xty;
    
    /* free new regression utility */
    delete_BayesReg(breg_new);
    delete_matrix(Xp_new);
    if(DiXp_new) delete_matrix(DiXp_new);
  }
}


/*
 * NewCol:
 *
 * grab X[,col] and apply the right transformation
 * and return it as a vector -- possibly taking DA rows
 * into account */

double* Blasso::NewCol(unsigned int col)
{
  /* sanity check */
  assert(col < M);
  assert(pb[col] == false);

  /* allocate the new (colu mn) vector */
  double *xnew = new_vector(n);
  unsigned int k, ell;
  k = ell = 0;
  unsigned int *R2 = NULL;
  if(R) R2 = R->R2[M];

  /* for each row in Xorig */
  for(unsigned int i=0; i<N; i++) {

    /* skip this row if Rt[M] == 2 */
    if(R2 && ell < R->n2[M] && R2[ell] == i) { ell++; continue; } 

    /* copy the col-th column */
    xnew[k] = Xorig[i][col] - Xmean[col];
    if(normalize) xnew[k] /= Xnorm_scale * Xnorm[col];
    k++;
  }
  assert(k == n);

  /* return the new column */
  return xnew;
}


/*
 * DataAugment:
 *
 * get the DA samples from Xorig and copy them into
 * Xp
 */

void Blasso::DataAugment(void)
{
  /* check if there is any missingness patterns to worry about */
  if(!R) return;

  /* loop over the columns */
  unsigned int changes = 0;
  for(unsigned int i=0; i<m; i++) {
    if(R->n2[pin[i]] == 0) continue;

    /* get the new indices in Xp which adjusts for rows omitted due to
       missing entries in Y; could speed up by passing n as maximum */
    int *R2 = adjust_elist(R->R2[pin[i]], R->n2[pin[i]], R->R2[M], R->n2[M]);

    /* loop over the (adjusted) rows listed in R2 */
    for(unsigned int j=0; j<R->n2[pin[i]]; j++) {

      if(R2[j] >= (int) n) break; /* don't go beyond the last row in Xp */
      else if(R2[j] == -1) continue;  /* skip rows that are skipped in Y */

      /* copy from Xorig to Xp after centering and normallization */
      Xp[R2[j]][i+icept] = Xorig[R->R2[pin[i]][j]][pin[i]] - Xmean[pin[i]];
      if(normalize) Xp[R2[j]][i+icept] /= Xnorm_scale * Xnorm[pin[i]];
      if(this->DiXp) this->DiXp[R2[j]][i+icept] = Xp[R2[j]][i+icept]/omega2[R2[j]];

      /* increment the changes counter */
      changes++;
    }

    /* clean up */
    free(R2);
  }

  /* need to update the regression stuff if there have been changes */
  if(changes > 0) {
    if(XtY) {
      double **DiXp = this->DiXp;
      if(DiXp == NULL) DiXp = Xp;
      linalg_dgemv(CblasNoTrans,m+icept,n,1.0,DiXp,
		   m+icept,this->Y,1,0.0,XtY,1);
    }
    if(!Compute(true)) error("ill-posed regression in DataAugment");
  }
  /* MAY BE THAT LIGHTER INIT DOABLE WITHOUT reinit=true ABOVE */
}


/*
 * adjust_elist:
 *
 * list l1 is an exclusion list of length n1, and each entry should be
 * should be reduced by the number of elements in exclusion list l2
 * of length n2 that are less than it.  The resulting list (with
 * reduced entries) is returned.  Where l1 and l2 have commen entries
 * and -1 is put
 */

int *adjust_elist(unsigned int *l1, const unsigned int n1, unsigned int *l2,
		  const unsigned int n2)
{
  int *l1_dup = new_dup_ivector((int*) l1, n1);

  for(unsigned int j=0; j<n2; j++) {
    for(unsigned int k=0; k<n1; k++) {
      if(l1[k] == l2[j]) l1_dup[k] = -1;
      else if(l1[k] > l2[j]) (l1_dup[k])--;
    }
  }

  return(l1_dup);
}


/*
 * Compute:
 *
 * compute the (mle) linear parameters Bmu (the mean)
 * A (the correllation matrix) and its inverse and
 * calculate the product t(Bmu) %*% A %*% Bmu -- see
 * the corresponding C function below
 */

bool Blasso::Compute(const bool reinit)
{
  /* do nothing if no betas */
  if(m+icept == 0) return true;

  /* possibly re-initialize if something has changed about Xp */
  if(reinit) init_BayesReg(breg, m+icept, n, Xp, DiXp);

  /* acutally do the linear algebra calculations */
  bool ret = compute_BayesReg(m+icept, XtY, tau2i, lambda2, s2, breg); 

#ifdef DEBUG
  if(YtY - breg->BtAB <= 0) { 
    myprintf(stdout, "YtY=%.20f, BtAB=%.20f\n", YtY, breg->BtAB);
    assert(0);
  }
#endif
  /* return code for success or failure */
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
#ifdef DEBUG
  if(info != 0) assert(0);
#else
  if(info != 0) return false;
#endif

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
  if(m+icept == 0) return;

  /* draw the beta vector */
  draw_beta(m+icept, beta, breg, s2, rn);
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
  lratio -= 0.5*mdiff*log(tau2);
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
  shape += (n-1.0)/2.0;
  if(reg_model == OLS) shape -= m/2.0;
  assert(shape > 0.0);
  
  /* rate = (X*beta-hat - Y)' (X*beta-hat - Y) / 2 + B-hat'DB-hat / 2*/
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
  double sums2 = 0.0;
  if(omega2) for(unsigned int i=0; i<n; i++) 
	       sums2 += resid[i]*resid[i]/omega2[i];
  else sums2 = sum_fv(resid, n, sq);

  /* BtDB = beta'D beta/tau2 as long as lambda != 0 */
  double BtDiB;
  if(m+icept > 0 && (reg_model == LASSO || reg_model == HORSESHOE)) {
    dupv(BtDi, beta, m);
    if(tau2i) scalev2(BtDi, m+icept, tau2i);
    else scalev(BtDi, m+icept, 1.0/lambda2);
    BtDiB = linalg_ddot(m+icept, BtDi, 1, beta, 1);
  } else BtDiB = 0.0;
    
  /* shape = (n-1)/2 + m/2 */
  double shape = a;
  if(reg_model != OLS) shape += (n-1)/2.0 + (m+icept)/2.0;
  else shape += (n-1)/2.0;
  
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
 * conditional on beta and sigma2
 */

void Blasso::DrawTau2i(void)
{
  double l_numer, l_mup;

  /* special case where we're not actually doing lasso */
  if(m == 0) return;

  /* sanity checks */
  assert(lambda2 > 0 && tau2i != NULL);

  /* special kludgy version of horseshoe */
  if(reg_model == HORSESHOE) UpdateLambdaCPS(m, beta, lambda2, s2, tau2i);
  else { /* regular lasso update of latents */
    assert(reg_model == LASSO);

    /* part of the mu parameter to the inv-gauss distribution */
    l_numer = log(lambda2) + log(s2);
    
    for(unsigned int j=icept; j<m+icept; j++) {
      
      /* the rest of the mu parameter */
      l_mup = 0.5*l_numer - log(fabs(beta[j])); 
      
      /* sample from the inv-gauss distn */
      tau2i[j] = rinvgauss(exp(l_mup), lambda2);    
      
      /* check to make sure there were no numerical problems */
      if(tau2i[j] <= 0) {
#ifdef DEBUG
	myprintf(stdout, "j=%d, m=%d, n=%d, l2=%g, s2=%g, beta=%g, tau2i=%g\n", 
		 j-icept, m, n, lambda2, s2, beta[j], tau2i[j]);
#endif
	tau2i[j] = 0;
      }
    }
  }

  /* Pool with DrawOmega2 and call Compute outside */
 }


/*
 * DrawOmega2:
 *
 * Gibbs draw for the Omega2 n-vector (latent variables) for the
 * Student-t implementation by scale mixtures, conditional on beta
 * and sigma2
 */

void Blasso::DrawOmega2(void)
{
  assert(DiXp && omega2 != NULL);

  double shape = 0.5*(nu+1.0);
  for (unsigned int i=0; i<n; i++)	{ 
    double scale = 0.5*(nu+sq(resid[i])/s2); 
    omega2[i] = 1.0/rgamma(shape,1.0/scale);
    assert(!isinf(omega2[i]));
  }

  /* update XtY, which will change when omega2 has changed */
  UpdateXY();

  /* pool with DrawTau2i and call Compute outside */
}


/*
 * DrawNu:
 *
 * draw the degrees of freedom parameter, nu, based on the
 * current omega2s
 */

void Blasso::DrawNu(void)
{
  /* sanity checks */
  assert(omega2 && theta > 0);

  /* calculate eta */
  double eta = theta;
  for (unsigned int i=0; i<n; i++) 
    eta += 0.5*(log(omega2[i])+1.0/omega2[i]);

  /* use rejection sampling */
  nu = draw_nu_reject(n, eta, theta);
  // nu = draw_nu_mh(nu, n, eta);
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
  if(reg_model == LASSO || reg_model == HORSESHOE) { /* lasso */
   
    /* sanity check */
    if(tau2i == NULL) assert(m+icept == 0);

    /* special kludgy version of horseshoe */  
    if(reg_model == HORSESHOE) UpdateTauCPS(m, beta, tau2i, s2, &lambda2);
    else { /* regular LASSO update of lambda2 */

      /* set up gamma distribution parameters */
      double shape = (double) m + r;
      double rate = 0.0;
      for(unsigned int j=icept; j<m+icept; j++) {
	if(tau2i[j] == 0) {shape--; continue;}  /* for numerical problems */
	rate += 1.0/tau2i[j];
      }
      rate = rate/2.0 + delta;
      
      /* draw from a gamma distribution */
      lambda2 = rgamma(shape, 1.0/rate);
    }

  } else { /* ridge */

    /* no lambda2 parameter draws for RIDGE when m == 0 */
    if(m == 0) { assert(lambda2 == 0); return; }

    /* sanity check */
    assert(tau2i == NULL && reg_model != OLS);
    
    /* set up Inv-Gamma distribution parameters */
    double BtB = linalg_ddot(m+icept, beta, 1, beta, 1);
    double shape = (double) (m+icept + r)/2.0;
    double scale = (BtB/s2 + delta)/2.0;

    /* draw from an Inv-Gamma distribution */
    lambda2 = 1.0/rgamma(shape, 1.0/scale);

    /* lambda2 has changed so need to update beta params */
    if(!Compute(false) || BtB/s2 <= 0) 
      error("ill-posed regression in DrawLambda2, BtB=%g, s2=%g, m=%d",
	    BtB, s2, m);
  }
}


/*
 * DrawPi:
 *
 * draw from the posterior distribution of the pi parameter
 * which governs the Binomial prior on the model order;
 * there may be nothing to do if this prior is fixed (not 
 * hierarchical) and/or uniform
 */

void Blasso::DrawPi(void)
{
  /* do nothing if pi is fixed */
  if(mprior[1] == 0) return;

  /* sanity check */
  assert(mprior[0] != 0);

  /* set the new pi */
  pi = rbeta(mprior[0] + (double)m, mprior[1] + (double)(Mmax-m));
}


/*
 * logPosterior:
 *
 * calculate the log posterior of the Bayesian lasso
 * model with the current parameter settings -- up to
 * an addive constant of proportionality (on the log scale)
 */

void Blasso::logPosterior(void)
{
  double *tau2i = this->tau2i;
  
  /* calculate the log likelihood */
  lpost = llik = log_likelihood(n, resid, s2, nu);//, omega2);

  /* calculate the likelihood under the Normal model if we are in the 
     Student-t model */
  if(omega2) llik_norm = log_likelihood(n, resid, s2, 1e300*1e300);
  else llik_norm = llik;
  
  /* calculate the log prior */
  lpost += log_prior(n, m+icept, beta, s2, tau2i, reg_model == HORSESHOE,
		     lambda2, omega2, nu, a, b, r, delta, theta, Mmax, 
		     pi, mprior);

  /* the code below is not in log_prior because of m and icept */

  /* add in the model prior */
  lpost += lprior_model(m, Mmax, pi);
  assert(!isinf(lpost));
  
  /* add in the model order probability */
  if(mprior[1] != 0 && pi != 0) 
    lpost += dbeta(pi, mprior[0] + (double)m, mprior[1] + (double)(Mmax-m), 1);
  assert(!isinf(lpost));
}


/*
 * log_likelihood
 *
 * calculate the log likelihood of the Bayesian lasso
 * model with the current parameter settings 
 *
 * passes the log likelihood back via llik, if not NULL
 */

double log_likelihood(const unsigned int n, double *resid, const double s2,
		      const double nu)//, double *omega2)
{
  /* for summing in the (log) posterior */
  double llik = 0.0;

  /* calculate the likelihood prod[N(resid | 0, s2)] in log space */
  double sd = sqrt(s2);	
  unsigned int i;
  // if(!omega2) 
  if(isinf(nu)) for(i=0; i<n; i++) llik += dnorm(resid[i], 0.0, sd, 1);
  // else for(i=0; i<n; i++) llik += dnorm(resid[i], 0.0, sd*sqrt(omega2[i]), 1);
  else for(i=0; i<n; i++) llik += dt(resid[i]/sd, nu, 1);
  assert(!isinf(llik));
 
   /* return the log likelihood */
  return llik;
}


/*
 * log_prior
 *
 * calculate the log prior of the Bayesian lasso
 * model with the current parameter settings -- up to
 * an addive constant of proportionality (on the log scale)
 */

double log_prior(const unsigned int n, const unsigned int m, double *beta, 
		 const double s2, double *tau2i, bool hs, const double lambda2, 
		 double *omega2, const double nu, const double a, const double b, 
		 const double r, const double delta, const double theta, 
		 const unsigned int Mmax, double pi, double *mprior)
{
  /* for summing in the (log) posterior */
  double lprior = 0.0;
  double sd = sqrt(s2);

  /* add in the prior for beta */
  if(tau2i) { /* under the lasso */
    for(unsigned int i=0; i<m; i++) 
      if(tau2i[i] > 0)
	lprior += dnorm(beta[i], 0.0, sd*sqrt(1.0/tau2i[i]), 1);
  } else if(lambda2 > 0) { /* under ridge regression */
    double lambda = sqrt(lambda2);
    for(unsigned int i=0; i<m; i++)
      lprior += dnorm(beta[i], 0.0, sd*lambda, 1);
  } /* nothing to do under flat/Jeffrey's OLS prior */
  assert(!isinf(lprior));

  /* add in the prior for s2 */
  if(a != 0 && b != 0) 
    lprior += dgamma(1.0/s2, 0.5*a, 2.0/b, 1);
  else lprior += 0.0 - log(s2);  /* Jeffrey's */

  /* add in the prior for tau2 */
  if(tau2i && lambda2 != 0)
    if(hs) lprior += LambdaCPS_lprior(m, tau2i, lambda2); /* under horseshoe */
    else { /* otherwise is exponential for lasso */
      for(unsigned int i=0; i<m; i++)
	if(tau2i[i] > 0) lprior += dexp(1.0/tau2i[i], 2.0/lambda2, 1);
    }
  assert(!isinf(lprior));
  
  /* add in the lambda prior */
  if(tau2i) { /* lasso */
    if(lambda2 != 0 && r != 0 && delta != 0) 
      if(hs) lprior += TauCPS_lprior(lambda2); /* under horseshoe */
      else lprior += dgamma(lambda2, r, 1.0/delta, 1); /* or is Gamma for lasso */
  } else if(lambda2 != 0) { /* ridge */
    if(r != 0 && delta != 0) 
      lprior += dgamma(1.0/lambda2, r, 1.0/delta, 1); /* is Inv-gamma */
    else lprior += 0.0 - log(lambda2); /* Jeffrey's */
  }
  assert(!isinf(lprior));

  /* add in the Student-t prior */
  if(omega2) { /* add in the components of omega2 */
    /* for(unsigned int i=0; i<n; i++) {
      lprior += dgamma(1.0/omega2[i], 0.5*nu, 2.0/nu, 1);
      } */

    /* add in the prior for nu */
    assert(theta > 0);
    lprior += dexp(nu, 1.0/theta, 1);
  }

  /* return the log prior */
  return lprior;
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
 * FixedPi:
 *
 * return true if pi is held fixed 
 */

bool Blasso::FixedPi(void)
{
  if(!RJ || mprior[1] == 0) return true;
  else return false;
}


/*
 * TErrors:
 *
 * return true if scale mixtures are being used to
 * implement Student-t errors in this regression
 */

bool Blasso::TErrors(void)
{
  if(omega2) {
    assert(theta > 0);
    return true;
  } else return false;
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

  /*myprintf(stdout, "add_col:\n");
  myprintf(stdout, "\tin: ");
  printIVector(pin, m, stdout);
  myprintf(stdout, "\tout: ");
  printIVector(pout, M-m, stdout);*/
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

  /* myprintf(stdout, "remove_col:\n");
  myprintf(stdout, "\tin: ");
  printIVector(pin, m, stdout);
  myprintf(stdout, "\tout: ");
  printIVector(pout, M-m, stdout); */
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
    
    if(reg_model == LASSO || reg_model == HORSESHOE) return 2;  /* rjlasso */
    else if(reg_model == RIDGE) return 3;  /* rjridge */
    else return 4;

  } else { /* no RJ */

    if(reg_model == LASSO || reg_model == HORSESHOE) return 5; /* lasso */
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
 * by taking RJ and LASSO/HORSESHOE into account -- and
 * taking the Student-t latents into account
 */

unsigned int Blasso::Thin(const double thin)
{
  unsigned int adjusted_thin = 0;

  /* adjust the thinning level for lasso latents */
  if(RJ || reg_model == LASSO || reg_model == HORSESHOE) 
    adjusted_thin = (unsigned) ceil(thin*Mmax);
  else if(reg_model == RIDGE) adjusted_thin = (unsigned) ceil(thin*2);
  
  /* adjust the thinning level for the Student-t latents */
  if(omega2) adjusted_thin += (unsigned) ceil(thin*n);

  /* make sure thin is positive */
  if(adjusted_thin == 0) adjusted_thin++;
  return adjusted_thin;
}


/*
 * lprior_model:
 *
 * calculate the log prior probability of a regression model
 * with m covariates which is either Binomial(m|Mmax,pi)
 * or is uniform over 0,...,Mmax
 */

double lprior_model(const unsigned int m, const unsigned int Mmax, 
		    double pi)
{
  assert(pi >= 0 && pi <= 1);
  if(Mmax == 0 || pi == 0.0 || pi == 1.0) return 0.0;
  
  /* myprintf(stdout, "m=%d, Mmax=%d, pi=%g, lp=%g\n", 
     m, Mmax, pi, dbinom((double) m, (double) Mmax, pi, 1));  */

  return dbinom((double) m, (double) Mmax, pi, 1);
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
 * matrix decomposed cov_chol (n x n) 
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
double  **omega2_mat;
Blasso *blasso = NULL;
int blasso_seed_set;

void blasso_R(int *T, int *thin, int *M, int *n, double *X_in, 
	      double *Y, double *lambda2, double *mu, int *RJ, 
	      int *Mmax, double *beta, int *m, double *s2, 
	      double *tau2i, int *hs, double *omega2, double *nu, double *pi, 
	      double *lpost, double *llik, double *llik_norm, double *mprior, 
	      double *r, double *delta, double *a, double *b, double *theta, 
	      int *rao_s2, int *normalize, int *verb)
{
  /* sanity check */
  assert(*T > 1);

  /* copy the vector input X into matrix form */
  X = new_matrix_bones(X_in, *n, *M);

  /* get the random number generator state from R */
  GetRNGstate(); blasso_seed_set = 1;

  /* initialize a matrix for beta samples */
  beta_mat = new_matrix_bones(beta, *T, *M);

  /* initialize a matrix for tau2i samples */
  if(tau2i != NULL) { /* for lasso */
    tau2i_mat = new_matrix_bones(&(tau2i[*M]), (*T)-1, *M);
  } else tau2i_mat = NULL; /* for ridge or OLS */

  /* initialize a matrix for omega2 samples */
  if(omega2 != NULL) { /* for lasso */
    omega2_mat = new_matrix_bones(&(omega2[*n]), (*T)-1, *n);
  } else omega2_mat = NULL; /* not using scale mixture Student-t */

  /* starting and sampling lambda2 if not null */
  double lambda2_start = 0.0;
  double *lambda2_samps = NULL;
  if(lambda2 != NULL) {
    lambda2_start = lambda2[0];
    lambda2_samps = &(lambda2[1]);
  }

  /* extract nu starting value if relevant */
  double nu_start = 0.0;
  if(nu) nu_start = *nu;

  /* create a new Bayesian lasso regression */
  blasso =  new Blasso(*M, *n, X, Y, (bool) *RJ, *Mmax, beta_mat[0], 
		       lambda2_start, s2[0], tau2i, (bool) *hs, omega2, 
		       nu_start, mprior, *r, *delta, *a, *b, *theta, 
		       (bool) *rao_s2, (bool) *normalize, *verb);

  /* part of the constructor which could fail has been moved outside */
  blasso->Init();

  /* Gibbs draws for the parameters */
  blasso->Rounds((*T)-1, *thin, &(mu[1]), &(beta_mat[1]), &(m[1]), 
		 &(s2[1]), tau2i_mat, lambda2_samps, omega2_mat, 
		 &(nu[1]), &(pi[1]), &(lpost[1]), &(llik[1]), 
		 &(llik_norm[1]));

  delete blasso;
  blasso = NULL;

  /* give the random number generator state back to R */
  PutRNGstate(); blasso_seed_set = 0;

  /* clean up */
  free(X); X = NULL;
  free(beta_mat); beta_mat = NULL;
  if(tau2i_mat) { free(tau2i_mat); tau2i_mat = NULL; }
  if(omega2_mat) { free(omega2_mat); omega2_mat = NULL; }
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

  /* free the matrix representations of the inputs */
  if(X) { free(X); X = NULL; }
  if(beta_mat) { free(beta_mat); beta_mat = NULL; }
  if(tau2i_mat) { free(tau2i_mat); tau2i_mat = NULL; }
  if(omega2_mat) { free(omega2_mat); omega2_mat = NULL; }

  /* deal with the seed */
  if(blasso_seed_set) { PutRNGstate(); blasso_seed_set = 0; }
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


/*
 * adjust_ilist_R:
 *
 * for testing the adjust_ilist function
 */

void adjust_elist_R(int *l1, int *n1, int *l2, int *n2, int *l1_out)
{
  int *l1_dup = adjust_elist((unsigned*) l1, (unsigned) *n1, 
			     (unsigned*) l2, (unsigned) *n2);
  dupiv(l1_out, l1_dup, (unsigned) *n1);
  free(l1_dup);
}
}
