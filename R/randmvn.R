#******************************************************************************* 
#
# Estimation for Multivariate Normal Data with Monotone Missingness
# Copyright (C) 2007, University of Cambridge
# 
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Questions? Contact Robert B. Gramacy (bobby@statslab.cam.ac.uk)
#
#*******************************************************************************


## randmvn:
##
## returns N random (x) samples from a randomly generated normal
## distribution via (mu) and (S), also returned.  mu is a
## standard normal vector of d means, and S is an inverse-Wishart
## renerated (dxd) covariance matrix with d+2 degrees of freedom
## and an identity centering matrix (mean)

'randmvn' <-
function(N, d)
  {
    ## check N
    if(length(N) != 1 || N < 0) stop("N should be a nonnegative integer")

    ## check d
    if(length(d) != 1 || N <= 0) stop("d should be a positive integer")
    
    ## load mvtnorm
    if(require(mvtnorm, quietly=TRUE) == FALSE)
      stop("this function requires library(mvtnorm)")

    ## generate a coavariance matrix and mean vector
    S <- solve(rwish(d+2, diag(d)))
    mu <- rnorm(d)

    ## don't draw if N=0
    if(N == 0) return(list(mu=mu, S=S))

    ## draw N samples from the MVN
    x <- rmvnorm(N, mu, S)#, method="chol")
    return(list(x=x, mu=mu, S=S))
  }
