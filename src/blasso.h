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


#ifndef __BLASSO_H__
#define __BLASSO_H__ 

#include <fstream>
#include "ustructs.h"

typedef enum REG_MODEL {LASSO=901,OLS=902,RIDGE=903,FACTOR=904,HORSESHOE=905,NG=906} REG_MODEL;
typedef enum MAT_STATE {NOINIT=1001,COV=1002,CHOLCOV=1003} MAT_STATE;

 /* regression utility structure */
typedef struct bayesreg
{
  unsigned int m;           /* dimension of these matrices and vectors */
  double *XtX_diag;         /* diagonal entries of XtX */
  double **A;               /* inverse of Vb unscaled by s2 */
  double **A_chol;          /* utility for inverting A via cholesky */
  double **Ai;              /* inverse of A */
  double ldet_Ai;           /* log(det(Ai)) */
  double *bmu;              /* posterior mean for beta */
  double BtAB;              /* in R: BtAB = t(bmu) %*% A %*% bmu */
  double *ABmu;             /* in R: ABmu = A %*% bmu */
  double **Vb;              /* posterior covariance matrix for beta Vb = s2*Ai */
  MAT_STATE Vb_state;       /* state of the Vb matrix */
} BayesReg;


/*
 * CLASS for the implementation of a Bayesian Lasso sampler
 * together with an R interface written in plain C
 */

class Blasso
{
 private:

  REG_MODEL reg_model;
  
  /* design matrix dimension */
  unsigned int M;           /* number of columns in Xorig */
  unsigned int N;           /* number of rows in Xorig */
  unsigned int n;           /* number of rows in Xp (== N if(Rmiss==NULL))*/
  unsigned int nf;          /* number of (initial) columns to treat as factors */

  /* the original design matrix and transformations */
  double **Xorig;           /* the (original) design matrix */
  bool normalize;           /* normalize X (when TRUE) */
  double *Xnorm;            /* normalization constants for the cols of X */
  double Xnorm_scale;       /* scaling factor for Xnorm normalization constants */
  double *Xmean;            /* mean of each column of X for centering */
  unsigned int ldx;         /* leading dimension of X and Xorig */
  bool copies;              /* indicaties whether the above are copies (TRUE) or
			       memory duplicates of X, Xnorm and Xorig, or just
			       pointers to memory allocated elsewere (FALSE) outside
			       this module */

  /* reversible-jump controlled columns of X in Xp */
  bool icept;               /* indicates if an intercept is desired in the model */
  bool EI;                  /* indicates if the intercept (I) is handled explicitly (E) */
  unsigned int m;           /* number of columns/rows of current XtX (breg->A) */
  bool RJ;                  /* indicated whether to do RJ moves or not */
  bool *pb;                 /* booleans indicating the m colns of X that are in use */
  int *pin;                 /* integer list of the columns of X that are in use */  
  int *pout;                /* integer list of the columns of X that are not in use */
  unsigned int Mmax;        /* maximum number of allowable columns in Xp */
  double **Xp;              /* (normd/centered) design matrix -- only cols in use */
                            /* in R syntax Xp = X[,pin] */
  double **DiXp;            /* inv(D_omega) %*% Xp for use with scale-mix Student-t */

  /* stuff to do with the response vector Y */
  double *Y;                /* the (centered) response vector */
  Rmiss *R;                 /* the missing data (DA) indicator vector */
  double Ymean;             /* the mean of the response vector used for centering */
  double *XtY;              /* in R syntax: t(X) %*% Y */
  double YtY;               /* in R syntax: t(Y) %*% Y, or SSy */
  double *resid;            /* in R syntax: Y - X %*% beta */

  /* model parameters */
  double lambda2;           /* the lasso penalty parameter */
  double s2;                /* the noise variance */
  double *tau2i;            /* inverse of diagonal of beta normal prior cov matrix */
  double *beta;             /* sampled regression coefficients, conditional on breg */
  double pi;                /* prior parameter p in Bin(m|M,p) for model order m
                               or indicates Unif[0,...,Mmax] when p=0 */
  double *omega2;           /* diagonal of the covariance matrix D of Y~N(Xb,s2*D) */
  double nu;                /* degrees of freedom parameter in Student-t model */
  double gam;               /* gamma parameter in the NG model */

  /* regressions utility structure(s) */
  BayesReg *breg;           /* matrices and vectors of regression quantities */

  /* prior parameters */
  double a;                 /* IG alpha (scale) prior parameter for s2 */
  double b;                 /* IG beta (inverse-scale) prior parameter for s2 */
  bool rao_s2;              /* integrate out beta when drawing s2 (when TRUE) */
  double mprior[2];         /* when mprior[1]==0 then pi=mprior[0], otherwise
                               this contains g & h of the beta prior for pi */
  double r;                 /* Gamma alpha (shape) prior parameter for lambda */
  double delta;             /* Gamma beta (scale) prior parameter for lambda */
  double theta;             /* Exp rate parameter to omegas for St errors */

  /* posterior probability evaluation */
  double lpost;             /* log posterior of parameters (not including mu) */
  double llik;              /* log likelihood component of the log posterior */
  double llik_norm;         /* as above, but stricly for the Normal model for BF */

  /* other useful vectors */
  double *rn;               /* vector for N(0,1) draws used to sample beta */
  double *Xbeta_v;          /* untility vector for unnorming beta coefficients */
  double *BtDi;             /* in R syntax: t(B) %*% inv(D) %*% B */
  
  /* printing */
  unsigned int verb;

 protected:

  /* parameter manipulation  */
  void GetParams(double *mu, double *beta, int *m, double *s2, double *tau2i, 
		 double *omega2, double *nu, double *lambda2, double *gam, 
		 double *pi) const;
  void InitIndicators(const unsigned int M, const unsigned int Mmax, 
		      double *beta, int *facts, const unsigned int nf);
  void InitPB(double *beta, int *facts, const unsigned int nf);
  void InitParams(const REG_MODEL reg_model, double *beta, double s2, 
		  double lambda2);
  void InitParams(double * beta, const double lambda2, const double s2, 
		  double *tau2i, bool hs, double *omega2, 
		  const double nu, const double gam);
  void InitRegress(void);
  void InitX(const unsigned int n, double **X, const bool normalize);
  void InitX(const unsigned int n, double **Xorig, Rmiss *R, 
	      double *Xnorm, const double Xnorm_scale,double *Xmean, 
	      const unsigned int ldx, const bool normalize);
  void InitY(const unsigned int n, double *Y);
  void UpdateXY(void);

  /* MCMC rounds */
  void Draw(const unsigned int thin, const bool fixnu);

  /* RJ moves */
  void RJmove(void);
  void RJdown(double q);
  void RJup(double q);
  void add_col(unsigned int i, unsigned int col);
  void remove_col(unsigned int i, unsigned int col);
  double* NewCol(unsigned int col);
  double ProposeTau2i(double *lpq_ratio);
  double UnproposeTau2i(double *lqp_ratio, unsigned int iin);


  /* Bayesian lasso sampling from the full conditionals */
  void DrawBeta(void);
  void DrawS2(void);
  void DrawS2Margin(void);
  void DrawTau2i(void);
  void DrawOmega2(void);
  void RJerrors(void);
  void DrawNu(void);
  bool Compute(const bool reinit);
  void DrawLambda2(void);
  void DrawGamma(void);
  void DrawPi(void);

  /* likelihood and posterior */
  void logPosterior();

 public:

  /* constructors and destructors */
  Blasso(const unsigned int m, const unsigned int n, double **X, double *Y,
	 const bool RJ, const unsigned int Mmax, double *beta, 
	 const double lambda2, const double s2, double *tau2i, const bool hs,
	 double *omega2, const double nu, const double gam, double *mprior, 
	 const double r, const double delta, const double a, const double b, 
	 const double theta, const bool rao_s2, const bool icept, 
	 const bool normalize, const unsigned int verb);
  Blasso(const unsigned int m, const unsigned int N, double **Xorig,
	 Rmiss *R, double *Xnorm, const double Xnorm_scale, double *Xmean, 
	 const unsigned int ldx, double *Y, const bool RJ, const unsigned int Mmax, 
	 double *beta_start, const double s2, const double lambda2_start, 
	 double *mprior, const double r, const double delta, const double theta,
	 const REG_MODEL reg_model, int *facts, const unsigned int nf, 
         const bool rao_s2, const unsigned int verb);
  ~Blasso();
  void Economize(void);

  /* initialization */
  void Init(void);

  /* access functions */
  REG_MODEL RegModel(void);
  bool UsesRJ(void);
  bool FixedPi(void);
  bool TErrors(void);

  /* MCMC sampling */
  void Rounds(const unsigned int T, const unsigned int thin, double *mu, 
	      double **beta, int *m, double *s2, double **tau2i, double *lambda2, 
	      double *gam, double **omega2, double *nu, double *pi, double *lpost, 
	      double *llik, double *llik_norm);
  void Draw(const double thin, const bool usenu, double *mu, double *beta, 
	    int *m, double *s2, double *tau2i, double *lambda2, double *gam, 
	    double *omega2, double *nu, double *pi, double *lpost, double *llik, 
	    double *llik_norm);
  void DataAugment(void);

    
  /* printing and summary information */
  void PrintParams(FILE *outfile) const;
  int Method(void);
  unsigned int Thin(const double thin);
  int Verb(void);
};


/* particular conditionals in C */
void draw_tau2i_lasso(const unsigned int m, double *tau2i, double *beta, 
		      double lambda2, double s2);
void draw_tau2i_ng(const unsigned int m, double *tau2i, double *beta, 
		   double lambda2, double gam, double s2);

/* random number generation */
void mvnrnd(double *x, double *mu, double **cov, double *rn, 
	    const unsigned int n);
double rinvgauss(const double mu, const double lambda);

/* for the inverse gamma distribution */
double Igamma_inv(const double a, const double y, const int lower, 
		  const int ulog);
double Cgamma(const double a, const int log);
double Rgamma_inv(const double a, const double y, const int lower, 
		  const int log);

/* log pdf */
double mvnpdf_log_dup(double *x, double *mu, double **cov, 
		      const unsigned int n);
double mvnpdf_log(double *x, double *mu, double **cov, 
		  const unsigned int n);
double log_determinant_chol(double **M, const unsigned int n);
double lprior_model(const unsigned int m, const unsigned int Mmax, 
		    double pi);

/* for modular reversible jump */
double mh_accep_ratio(unsigned int n, double *resid, double *x, double bnew, 
		      double t2i, double mub, double vb, double s2);
void draw_beta(const unsigned int m, double *beta, BayesReg* breg, 
	       const double s2, double *rn);
double log_likelihood(const unsigned int n, double *resid, const double s2,
		      const double nu);//, double *omega2);
double log_prior(const unsigned int n, const unsigned int m, const bool EI,
		 double *beta, const double s2, double *tau2i, 
		 const REG_MODEL reg_model, const double lambda2, double *omega2, 
		 const double nu, const double gam, const double a, const double b, 
		 const double r, const double delta, const double theta, 
		 const unsigned int Mmax, const double pi, double *mprior);

/* regression utility structure */
BayesReg* new_BayesReg(const unsigned int m, const unsigned int n, double **Xp, 
		       double **DiXp);
void delete_BayesReg(BayesReg* breg);
bool compute_BayesReg(const unsigned int m, double *XtY, double *tau2i, 
		      const double lambda2, const double s2, BayesReg *breg);
void refresh_Vb(BayesReg *breg, const double s2);
double beta_lprob(const unsigned int m, double *beta, BayesReg *breg);
double rj_betas_lratio(BayesReg *b1, BayesReg *b2, 
		       const double s2, const double tau2);
void alloc_rest_BayesReg(BayesReg* breg);
BayesReg* plus1_BayesReg(const unsigned int m, const unsigned int n,
			 BayesReg *old, double *xnew, double **Xp);
BayesReg* init_BayesReg(BayesReg *breg, const unsigned int m, 
			const unsigned int n, double **Xp, double **DiXp);

int *adjust_elist(unsigned int *l1, const unsigned int n1, unsigned int *l2,
		  const unsigned int n2);
#endif
