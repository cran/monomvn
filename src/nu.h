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

#ifndef __NU_H__
#define __NU_H__ 

double nustar_urr(const double x, const int n, const double eta);
void nustar_durr(const double x, double *fn, double *df, const int n, const double eta);
double nustar_urr_root(void funcd(const double, double*, double*, const int, const double), 
		       const int n, const double eta, const double x1, const double x2, 
		       const double xacc);
double draw_nu_reject(const unsigned int n, const double eta, const double theta);
double draw_nu_mh(const double nu_old, const unsigned int n, const double eta);
double nu_lpdf(const double nu, const unsigned int n, const double eta);
double unif_propose_pos(const double last, double *q_fwd, double *q_bak);

#endif
