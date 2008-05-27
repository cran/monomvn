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


#ifndef __HSHOE_H__
#define __HSHOE_H__ 

void UpdateLambdaCPS(int p, double* Beta, double lambda2PC, double sigma2, double* tau2iPC);
void UpdateTauCPS(int p, double *Beta, double *tau2iPC, double sigma2, double *lambda2PC);
double LambdaCPS_lprior(int m, double *tau2iPC, double lambda2PC);
double TauCPS_lprior(double lambda2PC);
double LambdaCPS_prior_draw(double lambda2PC);

#endif
