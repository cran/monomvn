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

typedef enum REG_MODEL {LASSO=901,OLS=902} REG_MODEL;

/*
 * CLASS for the implementation of a Bayesian Lasso sampler
 * together with an R interface written in plain C
 */

class Blasso
{
 private:

  REG_MODEL reg_model;
  
  /* inputs */
  unsigned int m;
  unsigned int n;
  double **X;
  double **Xorig;
  double *Xnorm;
  double *Y;
  double Ymean;
  double **XtX;
  double *XtY;
  double YtY;
  bool rao_s2;
  bool normalize;

  /* model parameters */
  double lambda2;
  double *beta;
  double s2;
  double a;
  double b;
  double *tau2i;
  double r;
  double delta;

  /* regression utility */
  double **Vb;
  double *bmu;
  double BtAB;
  double **A;
  double **A_util;
  double **Ai;
  double *Xbeta_v;
  double *BtDi;
  double *ABmu;
  double *rn;
  
  /* printing */
  unsigned int verb;

 protected:

  /* parameter manipulation  */
  void GetParams(double *beta, double *s2, double *tau2i, 
		 double *lambda2) const;
  void InitParams(const REG_MODEL reg_model);
  void InitParams(const double lambda2, const double s2, double *tau2i);
  void InitRegress(void);
  void InitXY(const unsigned int m, const unsigned int n, 
		      double **X, double *Y, const bool normalize);

  /* MCMC rounds */
  void Draw(const unsigned int thin);

 public:

  /* constructors and destructors */
  Blasso(const unsigned int m, const unsigned int n, double **X, 
	 double *Y, const double lambda2, const double s2, double *tau2i, 
	 const double r, const double delta, const double a,
	 const double b, const bool rao_s2, const bool normalize, 
	 const unsigned int verb);
  Blasso(const unsigned int m, const unsigned int n, double **X, 
	 double *Y, const double r, const double delta, 
	 const REG_MODEL reg_model, bool rao_s2, const unsigned int verb);
  ~Blasso();

  /* access functions */
  REG_MODEL RegModel(void);

  /* MCMC sampling */
  void Rounds(const unsigned int T, const unsigned int thin, 
	      double *lambda2, double *mu, double **beta, 
	      double *s2, double **tau2i);
  void Draw(const unsigned int thin, double *lambda2, double *mu, 
	    double *beta,  double *s2, double *tau2i);

  /* Bayesian lasso sampling from the full conditionals */
  void DrawBeta(void);
  void DrawS2(void);
  void DrawS2Margin(void);
  void DrawTau2i(void);
  bool ComputeBmuA(void);
  void DrawLambda2(void);
    
  /* printing */
  void PrintInputs(FILE *outfile) const;
  void PrintParams(FILE *outfile) const;
};

/* random number generation */
void mvnrnd(double *x, double *mu, double **cov, double *rn, const unsigned int n);
double rinvgauss(const double mu, const double lambda);

/* for the inverse gamma distribution */
double Igamma_inv(const double a, const double y, const int lower, const int ulog);
double Cgamma(const double a, const int log);
double Rgamma_inv(const double a, const double y, const int lower, const int log);

#endif
