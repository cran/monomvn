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


## blasso:
##
## function for sampling from the posterior distribution of the
## regression coefficients (beta) and variance (s2) under the
## Bayesian Lasso linear model of Park & Casella

'blasso' <-
function(X, y, T=100, thin=10, lambda2=1, s2=1, tau2i=rep(1, ncol(X)),
         r=1, delta=1, a=0, b=0, rao.s2 = TRUE, normalize=TRUE, verb=1)
  {
    ## dimensions of the inputs
    m <- ncol(X)
    n <- nrow(X)

    ## save the call
    cl <- match.call()
    
    ## check/conform arguments
    X <- as.matrix(X)
    y <- as.numeric(y)
    if(length(y) != nrow(X))
      stop("must have nrow(X) == length(y)")

    ## check T
    if(length(T) != 1 || T < 1)
      stop("T must be a scalar integer >= 1")

    ## check thin
    if(length(thin) != 1 || thin < 1)
      stop("thin must be a scalar integer >= 1")
    
    ## check lambda2
    if(length(lambda2) != 1 || lambda2 <= 0)
      stop("lambda2 must be a positive scalar")

    ## check s2
    if(length(s2) != 1 || s2 <= 0)
      stop("s2 must be a positive scalar")

    ## check tau2i
    if(length(tau2i) != ncol(X) || !all(tau2i > 0))
      stop("must have length(tau2i) == ncol(X) and all positive")

    ## check r
    if(length(r) != 1 || r <= 0)
      stop("r must be a positive scalar")

    ## check delta
    if(length(delta) != 1 || delta <= 0)
      stop("delta must be a positive scalar")

    ## check a
    if(length(a) != 1 || a < 0)
      stop("a must be a non-negative scalar")

    ## check b
    if(length(b) != 1 || b < 0)
      stop("b must be a non-negative scalar")

    ## check rao.s2
    if(length(rao.s2) != 1 || !is.logical(rao.s2))
      stop("rao.s2 must be a scalar logical")
    
    ## check normalize
    if(length(normalize) != 1 || !is.logical(normalize))
      stop("normalize must be a scalar logical")

    ## check normalize
    if(length(verb) != 1 || verb < 0)
      stop("verb must be non-negative a scalar integer")

    ## call the C routine
    r <- .C("blasso_R",
            T = as.integer(T),
            thin = as.integer(thin),
            m = as.integer(m),
            n = as.integer(n),
            X = as.double(t(X)),
            y = as.double(y),
            lambda2 = as.double(rep(lambda2,T)),
            mu = double(T),
            beta = double(m*T),
            s2 = as.double(rep(s2, T)),
            tau2i = as.double(rep(tau2i, T)),
            r = as.double(r),
            delta = as.double(delta),
            a = as.double(a),
            b = as.double(b),
            rao.s2 = as.integer(rao.s2),
            normalize = as.integer(normalize),
            verb = as.integer(verb),
            PACKAGE = "monomvn")

    ## copy the inputs back into the returned R-object
    r$X <- X
    r$y <- y

    ## turn the beta and tau2i vectors of samples into matrices
    r$beta <- matrix(r$beta, nrow=T, ncol=m, byrow=TRUE,
                     dimnames=list(NULL,paste("b.", 1:m, sep="")))
    r$tau2i <- matrix(r$tau2i, nrow=T, ncol=m, byrow=TRUE,
                      dimnames=list(NULL,paste("tau2i.", 1:m, sep="")))

    ## null-out redundancies
    r$n <- r$m <- r$verb <- NULL

    ## assign call and class
    r$call <- cl
    class(r) <- "blasso"
    
    return(r)
  }
