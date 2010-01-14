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

#include "rgig.h"
#include "matrix.h"
#include <math.h>
#include <assert.h>
#include <R.h>
#include <Rmath.h>


#define ZTOL sqrt(DOUBLE_EPS)

/* 
 * gig_gfn: 
 *
 * evaluate the function that we need to find the root 
 * of in order to in order to construct the optimal rejection
 * sampler for to obtain GIG samples
 */

double gig_gfn(double y, double m, double beta, double lambda)
{	
  double y2, g;
  y2 = y*y;
  g = 0.5 * beta * y2*y;
  g -= y2 * (0.5 * beta * m + lambda + 1.0);
  g += y * ((lambda - 1.0) * m - 0.5 * beta) + 0.5 * beta * m;
  return(g);
}

	
/*
 ************************************************************************
 *	    		    C math library
 * function ZEROIN - obtain a function zero within the given range
 *
 * Input
 *	double zeroin(ax,bx,f,tol)
 *	double ax; 			Root will be seeked for within
 *	double bx;  			a range [ax,bx]
 *	double (*f)(double x);		Name of the function whose zero
 *					will be seeked for
 *	double tol;			Acceptable tolerance for the root
 *					value.
 *					May be specified as 0.0 to cause
 *					the program to find the root as
 *					accurate as possible
 *
 * Output
 *	Zeroin returns an estimate for the root with accuracy
 *	4*EPSILON*abs(x) + tol
 *
 * Algorithm
 *	G.Forsythe, M.Malcolm, C.Moler, Computer methods for mathematical
 *	computations. M., Mir, 1980, p.180 of the Russian edition
 *
 *	The function makes use of the bissection procedure combined with
 *	the linear or quadric inverse interpolation.
 *	At every step program operates on three abscissae - a, b, and c.
 *	b - the last and the best approximation to the root
 *	a - the last but one approximation
 *	c - the last but one or even earlier approximation than a that
 *		1) |f(b)| <= |f(c)|
 *		2) f(b) and f(c) have opposite signs, i.e. b and c confine
 *		   the root
 *	At every step Zeroin selects one of the two new approximations, the
 *	former being obtained by the bissection procedure and the latter
 *	resulting in the interpolation (if a,b, and c are all different
 *	the quadric interpolation is utilized, otherwise the linear one).
 *	If the latter (i.e. obtained by the interpolation) point is 
 *	reasonable (i.e. lies within the current interval [b,c] not being
 *	too close to the boundaries) it is accepted. The bissection result
 *	is used in the other case. Therefore, the range of uncertainty is
 *	ensured to be reduced at least by the factor 1.6
 *
 ************************************************************************
 */

/* THIS FUNCTION HAS BEEN MODIFIED TO DEAL WITH GIG_GFN (extra args) */

double zeroin_gig(ax,bx,f,tol, m, beta, lambda)	/* An estimate to the root  */
double ax;				/* Left border | of the range	*/
double bx;  				/* Right border| the root is seeked*/
/* Function under investigation	*/
double (*f)(double x, double m, double beta, double lambda);	
double tol;				/* Acceptable tolerance	*/
double m;                               /* specific to gig_gfn */
double beta;                            /* specific to gig_gfn */
double lambda;                          /* specific to gig_gfn */
{
  double a,b,c;				/* Abscissae, descr. see above	*/
  double fa;				/* f(a)				*/
  double fb;				/* f(b)				*/
  double fc;				/* f(c)				*/

  a = ax;  b = bx;  fa = (*f)(a, m, beta, lambda);  fb = (*f)(b, m, beta, lambda);
  c = a;   fc = fa;

  for(;;)		/* Main iteration loop	*/
  {
    double prev_step = b-a;		/* Distance from the last but one*/
					/* to the last approximation	*/
    double tol_act;			/* Actual tolerance		*/
    double p;      			/* Interpolation step is calcu- */
    double q;      			/* lated in the form p/q; divi- */
  					/* sion operations is delayed   */
 					/* until the last moment	*/
    double new_step;      		/* Step at this iteration       */
   
    if( fabs(fc) < fabs(fb) )
    {                         		/* Swap data for b to be the 	*/
	a = b;  b = c;  c = a;          /* best approximation		*/
	fa=fb;  fb=fc;  fc=fa;
    }
    tol_act = 2*ZTOL*fabs(b) + tol/2;
    new_step = (c-b)/2;

    if( fabs(new_step) <= tol_act || fb == (double)0 )
      return b;				/* Acceptable approx. is found	*/

    			/* Decide if the interpolation can be tried	*/
    if( fabs(prev_step) >= tol_act	/* If prev_step was large enough*/
	&& fabs(fa) > fabs(fb) )	/* and was in true direction,	*/
    {					/* Interpolatiom may be tried	*/
	register double t1,cb,t2;
	cb = c-b;
	if( a==c )			/* If we have only two distinct	*/
	{				/* points linear interpolation 	*/
	  t1 = fb/fa;			/* can only be applied		*/
	  p = cb*t1;
	  q = 1.0 - t1;
 	}
	else				/* Quadric inverse interpolation*/
	{
	  q = fa/fc;  t1 = fb/fc;  t2 = fb/fa;
	  p = t2 * ( cb*q*(q-t1) - (b-a)*(t1-1.0) );
	  q = (q-1.0) * (t1-1.0) * (t2-1.0);
	}
	if( p>(double)0 )		/* p was calculated with the op-*/
	  q = -q;			/* posite sign; make p positive	*/
	else				/* and assign possible minus to	*/
	  p = -p;			/* q				*/

	if( p < (0.75*cb*q-fabs(tol_act*q)/2)	/* If b+p/q falls in [b,c]*/
	    && p < fabs(prev_step*q/2) )	/* and isn't too large	*/
	  new_step = p/q;			/* it is accepted	*/
					/* If p/q is too large then the	*/
					/* bissection procedure can 	*/
					/* reduce [b,c] range to more	*/
					/* extent			*/
    }

    if( fabs(new_step) < tol_act ) {	/* Adjust the step to be not less*/
      if( new_step > (double)0 )	/* than tolerance		*/
	new_step = tol_act;
      else
	new_step = -tol_act;
    }

    a = b;  fa = fb;			/* Save the previous approx.	*/
    b += new_step;  fb = (*f)(b, m, beta, lambda);  /* Do step to a new approxim. */
    if( (fb > 0 && fc > 0) || (fb < 0 && fc < 0) )
    {                 			/* Adjust c for it to have a sign*/
      c = a;  fc = fa;                  /* opposite to that of b	*/
    }
  }

}


/*
 * rgig:
 *
 * a C implementation of the general case of the R code for rgig from
 * the ghyp package on CRAN
 */

void rgig(const int n, const double lambda, const double chi, const double psi,
	  double *samps)
{
  double alpha, beta, m, upper, yM, yP, a, b, c, R1, R2, Y, b2, lm1, lm12;
  int i, need;

  /* baseline sanity check */
  if(chi < ZTOL && psi < ZTOL) {
    for(i=0; i<n; i++) samps[i] = -1.0;
    return;
  }
  
  /* sanity check for chi */
  if(chi < ZTOL) {
    if(lambda > 0) for(i=0; i<n; i++) samps[i] = rgamma(lambda, 2.0/psi);
    else for(i=0; i<n; i++) samps[i] = -1.0;
    return;
  }

  /* sanity check for psi */
  if(psi < ZTOL) {
    if(lambda < 0) for(i=0; i<n; i++) samps[i] = 1.0/rgamma(0.0-lambda, 2.0/chi);
    else for(i=0; i<n; i++) samps[i] = -1.0;
    return;
  }

  /*
   * begin code stolen from rgig in the ghyp package 
   */
  /* fprintf(stdout, "begin gam\n"); */

  alpha = sqrt(psi/chi);
  beta = sqrt(psi * chi);
  b2 = beta*beta;
  lm1 = lambda - 1.0;
  lm12 = lm1*lm1;
  
  m = (lm1 + sqrt(lm12 + b2))/beta;

  upper = m;
  while (gig_gfn(upper, m, beta, lambda) <= 0) { upper *= 2; }

  yM = zeroin_gig(0.0, m, gig_gfn, ZTOL, m, beta, lambda);
  yP = zeroin_gig(m, upper, gig_gfn, ZTOL, m, beta, lambda);

  a = (yP - m) * pow(yP/m, 0.5 * (lambda - 1));
  a *=  exp(-0.25 * beta * (yP + 1/yP - m - 1/m));
  b = (yM - m) * pow(yM/m, 0.5 * (lambda - 1));
  b *= exp(-0.25 * beta * (yM + 1/yM - m - 1/m));
  c = -0.25 * beta * (m + 1/m) + 0.5 * (lambda - 1) * log(m);

  for (i=0; i<n; i++) {
    need = 1;
    while (need) {
      R1 = runif(0.0, 1.0);
      R2 = runif(0.0, 1.0);
      Y = m + a * R2/R1 + b * (1 - R2)/R1;
      if (Y > 0) {
	if (-log(R1) >= -0.5 * (lambda - 1) * log(Y) + 
	    0.25 * beta * (Y + 1.0/Y) + c) {
	  need = 0;
	}
      }
    }
    samps[i] = Y/alpha;
  }

  /* fprintf(stdout, "end gam\n"); */
}


/* 
 * rgig_R:
 *
 * wrapper function for the .C call from R
 */

void rgig_R(int *n_in, double *lambda_in, double *chi_in, 
	    double *psi_in, double *samps_out)
{
  GetRNGstate();

  rgig(*n_in, *lambda_in, *chi_in, *psi_in, samps_out);

  PutRNGstate();
}
