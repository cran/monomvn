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


#define DEBUG

#include "nu.h"
#include "rhelp.h"
#include <math.h>
#include <assert.h>
#include <Rmath.h>


/* 
 * nustar_urr: 
 *
 * evaluate the function that we need to find the root 
 * of in order to minimize the unconditional rejection rate 
 * for proposing from Exp(x) to rejection sample from 
 * the full conditional for nu
 */

double nustar_urr(const double x, const int n, const double eta)
{	
  double dn = (double) n;
  return 0.5*dn*(log(0.5*x) + 1.0 - digamma(0.5*x)) + 1.0/x - eta;
}

	
/* 
 * nustar_durr: 
 * 
 * passes back the evaluation of the function we need to find
 * the root of to minimize the unconditional rejection rate 
 * nustarr_urr(x,n,eta) and its derivartive at x.
 */

void nustar_durr(const double x, double *fn, double *df, const int n, const double eta)
{   
  double dn = (double) n;
  *fn = nustar_urr(x, n, eta); 
  *df = 0.5*dn/x - 0.25*dn*trigamma(0.5*x) - 1.0/(x*x);
}


/*
 * nustar_urr_root:
 *
 * finds the root of the function needed to minimize the unconditional 
 * rejection rate for proposing from Exp(nustar) to rejection sample 
 * from the full conditional for nu -- this code is a adaptation of
 * the "rtsafe" function provided by W.H.Press et al., "Numerical 
 * Recipes in C++, The art of scientific computing", CUP 2002, 2nd ed.
 *
 * MAYBE CALL THIS FUNCTION root_safe_newton instead, except it seems
 * to take extra arguments (n and eta) which are particular to nustar
 * finding
 */

double nustar_urr_root 
(void funcd(const double, double*, double*, const int, const double), 
  const int n, const double eta, const double x1, const double x2, 
  const double xacc)
{	
  /* const int MAXIT=100; */   /* maximum allowed number of iterations */
  double df,dx,dxold,f,fh,fl,temp,xh,xl,rts;
	
  funcd(x1,&fl,&df,n,eta);
  funcd(x2,&fh,&df,n,eta);
  if ((fl > 0.0 && fh> 0.0) || (fl < 0.0 && fh < 0.0))
#ifdef DEBUG
    assert(0);
#else
    error("Root must be bracketed for bisection in rootsafenewton");
#endif
  if (fl == 0.0) return x1;
  if (fh == 0.0) return x2;
  if (fl < 0.0) {                /* orient the search so that f(x1)<0. */
    xl=x1;
    xh=x2;
  }	else {
    xh=x1;
    xl=x2;
  }
  rts=0.5*(x1+x2);               /* initialize the guess for root, */
  dxold=fabs(x2-x1);             /* the stepsize before last, */
  dx=dxold;                      /* and the last step. */
  funcd(rts,&f,&df,n,eta);
  /* for (j=0;j<MAXIT;j++) {  */ /* loop over allowed iterations */
  for (;;) {
    if ((((rts-xh)*df-f)*((rts-x1)*df-f) > 0.0) 
	|| (fabs(2.0*f) > fabs(dxold*df))) {    /* Bisect if Newton out of range */
      dxold=dx;
      dx=0.5*(xh-xl);
      rts=xl+dx;
      if (xl == rts) return rts;
    } else {
      dxold=dx;
      dx=f/df;
      temp=rts;
      rts-=dx;
      if (temp ==rts) return rts;
    }
    if (fabs(dx) < xacc) return rts;      /* convergence criterion */
    funcd(rts,&f,&df,n,eta);
    if (f < 0.0)
      xl=rts;
    else
      xh=rts;
  }
  /* error("Maximum number of iterations exceeded in rootsafenewton"); */
  return 0.0;
}


/*
 * draw_nu_reject:
 *
 * use proposals from an exponential distribution with
 * an optimal scale parameter (minimizing the unconditional
 * rejection rate) to sample from the full conditional of 
 * the degrees of freedom parameter to the Student-t distribution
 */

double draw_nu_reject(const unsigned int n, const double eta, 
		      const double theta)
{
  double x1, x2, f1, f2, nustar, u, nu, dn;
  unsigned int counter;

  /* bracketing for root finding */
  x1 = 0.5; x2 = 2;
  f1 = nustar_urr(x1, n, eta);
  f2 = nustar_urr(x2, n, eta);
  counter=0;	
  do {
    counter++;
    x1 = 0.5*x1; x2 = 2.0*x2;
    f1 = nustar_urr(x1, n, eta);
    f2 = nustar_urr(x2, n, eta);
  } while(f1*f2 >= 0.0 && counter<100);

  /* check that we've actually been able to bracket the root */
  if (counter==100) warning("draw_nu_reject: theta might be too high");
	
  /* finding the root */
  nustar = nustar_urr_root(nustar_durr,n,eta,x1,x2,1e-7);

  /* rejection sampling for nu */
  dn = (double) n;
  do{ /* until acceptance */
    u = unif_rand();    
    //nu = rexp(1.0/nustar);
    nu = rexp(nustar);
  } while(log(u) >= (dn*(0.5*nu)*log(0.5*nu) - (0.5*dn*nustar)*log(0.5*nustar)
		     + dn*lgammafn(0.5*nustar) - dn*lgammafn(0.5*nu)
		     + (nu-nustar)*(1.0/nustar-eta))); 
  
  /* done */
  return(nu);
 }


/*
 * unif_propose_pos:
 *
 * propose a new positive "ret" based on an old value "last"
 * by proposing uniformly in [3last/4, 4last/3], and return
 * the forward and backward probabilities;
 */

#define PNUM 3.0
#define PDENOM 4.0

double unif_propose_pos(const double last, double *q_fwd, double *q_bak)
{
  double left, right, ret;

  /* propose new d, and compute proposal probability */
  left = PNUM*last/(PDENOM);
  right = PDENOM*last/(PNUM);
  assert(left > 0 && left < right);
  ret = runif(left, right);
  *q_fwd = 1.0/(right - left);

  /* compute backwards probability */
  left = PNUM*ret/(PDENOM);
  right = PDENOM*ret/(PNUM);
  assert(left >= 0 && left < right);
  *q_bak = 1.0/(right - left);
  assert(*q_bak > 0);

  /* make sure this is reversible */
  assert(last >= left && last <= right);

  /* if(ret > 10e10) {
    warning("unif_propose_pos (%g) is bigger than max", ret);
    ret = 10;
    } */
  assert(ret > 0);
  return ret;
}

double nu_lpdf(const double nu, const unsigned int n, const double eta)
{
  double dn = (double) n;
  return 0.5*(dn*nu)*log(0.5*nu) - dn*lgamma(0.5*nu) - eta*nu;
}

double draw_nu_mh(const double nu_old, const unsigned int n, const double eta) 
{
  double qf, qb, nu, alpha, u;
  
  //nu = unif_propose_pos(nu_old-1.0, &qf, &qb) + 1.0;
  nu = unif_propose_pos(nu_old, &qf, &qb);
  alpha = exp(nu_lpdf(nu, n, eta) - nu_lpdf(nu_old, n, eta));
  MYprintf(MYstdout, "nu_old=%g, nu=%g, alpha=%g, qratio=%g", nu_old, nu, alpha, qb/qf);
  u = unif_rand();
  MYprintf(MYstdout, " u=%g", u);
  if(u < alpha*qb/qf) { MYprintf(MYstdout, " accept\n"); return nu; }
  else { MYprintf(MYstdout, " reject\n"); return nu_old; }
}
