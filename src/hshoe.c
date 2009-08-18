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


#include "hshoe.h"
#include <Rmath.h>
#include <math.h>


/* 
 * UpdateLambdaCPS:
 *
 * Carvalho, Polson & Scott (CPS) code that that samples
 * from the horseshoe using the Park & Casella (PC) 
 * parameterization (employed by blasso) to sample from
 * the Lambda parameter, which we call tau2i.
 *
 * Thanks to James Scott for this code
 */

void UpdateLambdaCPS(int p, double* Beta, double lambda2PC, 
		     double sigma2, double* tau2iPC) 
{ 
  double mu; 
  double kappa, u, upper, rate, bound, tauCPS, lambdaCPS; 
  int ltail, logflag, j; 

  ltail=1; logflag=0; 
  tauCPS = sqrt(1/lambda2PC);
  for(j=0; j<p; j++) 
    {
      lambdaCPS = 1.0/tauCPS;
      lambdaCPS *= 1.0/sqrt(tau2iPC[j]);
      mu = sqrt(1.0/sigma2) * (1.0/tauCPS) * Beta[j]; 
      kappa = 1.0/(lambdaCPS * lambdaCPS); 
      kappa = 1.0/(1.0 + kappa); 
      u = runif(0.0, kappa); 
      bound = (1.0-u)/u; 
      rate = mu*mu/2.0; 
      rate = 1.0/rate; 
      upper = pexp(bound, rate, ltail, logflag); 
      u = runif(0,upper); 
      kappa = qexp(u, rate, ltail, logflag); 
      rate = sqrt(1.0/kappa); 
      lambdaCPS = rate;
      /* tau2iPC[j] = pow(1/(lambdaCPS * tauCPS), 2.0); */
      tau2iPC[j] = 1.0/(lambdaCPS * tauCPS);
      tau2iPC[j] *= tau2iPC[j];
    } 
}
 

/* 
 * UpdateTauCPS:
 *
 * Carvalho, Polson & Scott (CPS) code that that samples
 * from the horseshoe using the Park & Casella (PC) 
 * parameterization (employed by blasso) to sample from
 * the Tau parameter, which we call lambda2.
 *
 * Thanks to James Scott for this code
 */

void UpdateTauCPS(int p, double *Beta, double *tau2iPC, 
		  double sigma2, double *lambda2PC) 
{ 
  double a, b, mytau, lambdaCPS;
  double eta, u, q, upper, scale, bound; 
  int ltail, logflag, j; 

  /* sample from the prior when p == 0 */
  if(p == 0) {
    mytau = rt(1.0);
    *lambda2PC = 1.0/(mytau*mytau);
    return;
  }

  mytau = *lambda2PC; mytau = sqrt(1.0/mytau);  
  ltail=1; logflag=0;
  b = 0.0;

  a = ((double)p+1.0)/2.0; 
  for(j = 0; j < p; j++) 
    { 
      lambdaCPS = 1.0/mytau;
      lambdaCPS *= 1.0/sqrt(tau2iPC[j]);
      /* b += pow(Beta[j],2.0) / (pow(lambdaCPS, 2.0) * sigma2); */
      b += (Beta[j]*Beta[j]) / ((lambdaCPS*lambdaCPS) * sigma2); 
    } 
  b = 0.5*b; 
 
  /* First sample u | eta */
  /* eta = 1.0/(pow(mytau, 2.0)); */
  eta = 1.0/((mytau*mytau)); 
  bound = 1.0/(1.0 + eta); 
  u = runif(0.0, bound); 
 
  /* Now sample eta | u */
  bound = (1.0-u)/u; 
  scale = 1/b; 
  upper = pgamma(bound, a, scale, ltail, logflag); 
  u = runif(0,upper); 
  eta = qgamma(u, a, scale, ltail, logflag); 
  q = sqrt(1.0/eta); 
  mytau = q;
  *lambda2PC = 1.0/(mytau*mytau);
} 
 

/*
 * LambdaCPS_lprior:
 *
 * (log) prior probability of the components of tau2i
 * under the horseshoe
 */

double LambdaCPS_lprior(int m, double *tau2iPC, double lambda2PC)
{
  double mytau, lambdaCPS, lprior;
  lprior = m*log(2.0);
  mytau = lambda2PC; mytau = sqrt(1.0/mytau);
  for(int j=0; j<m; j++) {
    lambdaCPS = 1.0/mytau;
    lambdaCPS *= 1.0/sqrt(tau2iPC[j]);
    lprior += dt(lambdaCPS, 1.0, 1);
  }
  return lprior;
}


/*
 * LambdaPCS_prior_draw:
 *
 * draw tau2 from the prior as defined by CPS
 * NOTE the draw is for tau2 not tau2i
 */

double LambdaCPS_prior_draw(double lambda2PC)
{
  double lambdaCPS, mytau;
  mytau = lambda2PC; mytau = sqrt(1.0/mytau);
  lambdaCPS = fabs(rt(1.0));
  return /*1.0/*/sqrt(mytau*lambdaCPS);
}


/*
 * TauCPS_lprior:
 *
 * (log) prior probability of lambda2
 * under the horseshoe
 */

double TauCPS_lprior(double lambda2PC)
{
  double mytau = 1.0/sqrt(lambda2PC);
  return log(2.0) + dt(mytau, 1.0, 1);
}


/* 
 * UpdateLambdaCPS_NEG:
 *
 * Carvalho, Polson & Scott (CPS) code that that samples
 * from the Normal Exponential Gamma prior using the 
 * Park & Casella (PC) parameterization (employed by blasso) 
 * to sample from the Lambda parameter, which we call tau2i.
 *
 * Thanks to James Scott for this code
 */

void UpdateLambdaCPS_NEG(int p, double a, double* Beta, 
			 double lambda2PC, double sigma2, 
			 double* tau2iPC)
{
  int j;
  double mu, z, u1, u2, lower, upper, rate, lbound, ubound, lambdaCPS;
  double tauCPS = sqrt(1.0/lambda2PC);

  for(j=0; j<p; j++) {

    lambdaCPS = 1.0/tauCPS;
    lambdaCPS *= 1.0/sqrt(tau2iPC[j]);
    mu = sqrt(1.0/sigma2) * (1.0/tauCPS) * Beta[j]; 
    z = 1.0/(lambdaCPS * lambdaCPS); 
    
    upper = pow(z + 1.0, -(a+1.0));
    u1 = runif(0.0, upper);
    upper = pow(z, a-0.5);
    u2 = runif(0.0, upper);
    
    /* Now we establish the upper and lower bounds 
       for the slice region of z */
    if(a >= 0.5) {
      lower = pow(u2, 1/(a-0.5));
      upper = pow(u1, -1.0/(a+1.0)) - 1;
    } else {
      lower = 0.0;
      upper = pow(u1, -1.0/(a+1.0)) - 1;
      rate = pow(u2, -a+0.5);
      if(rate < upper) upper = rate;
    }
    
    /* Now we express that slice region in terms of 
       the exponential cdf */
    rate = mu*mu/2.0;
    rate = 1.0/rate;
    if(lower == 0.0) lbound = 0.0;
    else lbound = pexp(lower, rate, 1, 0);
    ubound = pexp(upper, rate, 1, 0);
    
    /* Now we draw the uniform random variable and 
       use inverse CDF to get draw */
    u1 = runif(lbound,ubound);
    z = qexp(u1, rate, 1, 0);
    rate = sqrt(1.0/z);
    lambdaCPS = rate;
    /* tau2iPC[j] = pow(1/(lambdaCPS * tauCPS), 2.0); */
    tau2iPC[j] = 1.0 / lambdaCPS * tauCPS;
    tau2iPC[j] *= tau2iPC[j];
  }
}

