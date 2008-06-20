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
function(X, y, T=100, thin=10, RJ=TRUE, M=NULL, beta=NULL,
         lambda2=1, s2=1, tau2i=NULL, rd=c(2,0.1), ab=NULL,
         rao.s2=TRUE, normalize=TRUE, verb=1)
  {
    ## (quitely) double-check that blasso is clean before-hand
    blasso.cleanup()
    
    ## what to do if fatally interrupted?
    on.exit(blasso.cleanup())

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
    if(length(T) != 1 || T <= 1)
      stop("T must be a scalar integer > 1")

    ## check thin
    if(length(thin) != 1 || thin < 1)
      stop("thin must be a scalar integer >= 1")

    ## check RJ (reversible jump)
    if(length(RJ) != 1 || !is.logical(RJ))
      stop("RJ must be a scalar logical\n")

    ## check M or default
    if(is.null(M)) M <- as.integer(min(m, n-1))
    M <- as.integer(M)
    if(length(M) != 1 || M < 0 || M > m)
      stop("M must be a positive integer 0 <= M <= ncol(X)")
    if(!RJ && M != m) {
      M <- m
      warning("must have M=ncol(X) when RJ=FALSE", immediate.=TRUE)
    } 
    
    ## check beta or default
    if(is.null(beta)) beta <- rep(!RJ, m)
    if(length(beta) != m)
      stop("must have length(beta) == ncol(X)")

    ## check lambda2
    if(length(lambda2) != 1 || lambda2 < 0)
      stop("lambda2 must be a non-negative scalar")

    ## general big-p small-n problem -- must have lasso on
    if(lambda2 == 0 && m >= n)
      stop("big p small n problem; must have lambda2 > 0")
    
    ## check for a valid regression
    if(!RJ) {  ## when not doing reversible jump (RJ)
      if(lambda2 > 0 && any(beta == 0)) {
        warning("must start with non-zero beta when RJ=FALSE, using beta=1\n",
                immediate.=TRUE)
        beta  <- rep(!RJ, m)
      }
    } else if(sum(beta != 0) > M) { ## when doing RJ
      beta <- rep(0, m)
      warning("initial beta must have M or fewer non-zero entries",
              immediate.=TRUE)
    }

    ## check s2
    if(length(s2) != 1 || s2 <= 0)
      stop("s2 must be a positive scalar")

    ## check tau2i or default
    if(is.null(tau2i)) tau2i <- rep(lambda2!=0, m)
    if(length(tau2i) != m || (lambda2 && !all(tau2i > 0)))
      stop("must have length(tau2i) == ncol(X) and all > 0 when lambda2 > 0")
    tau2i[beta == 0] <- -1

    ## check r and delta (rd)
    if(length(rd) != 2 || any(rd <= 0))
      stop("rd must be a positive 2-vector")

    ## check ab or default
    if(is.null(ab)) {
      ab <- c(0,0)
      if(!RJ && lambda2 > 0 && m >= n) {
        ab[1] <- 3/2
        ab[2] <- Igamma.inv(ab[1], 0.99*gamma(ab[1]), lower=FALSE)*sum(y^2)
      } 
    }

    ## double check ab
    if(length(ab) != 2 || any(ab < 0))
      stop("ab must be a non-negative 2-vector")
    if(!RJ && m >= n && any(ab <= 0))
      stop("must have ab > c(0,0) when !RJ and ncol(X) >= length(y)")

    ## check rao.s2
    if(length(rao.s2) != 1 || !is.logical(rao.s2))
      stop("rao.s2 must be a scalar logical")
    
    ## check normalize
    if(length(normalize) != 1 || !is.logical(normalize))
      stop("normalize must be a scalar logical")

    ## check verb
    if(length(verb) != 1 || verb < 0)
      stop("verb must be non-negative a scalar integer")

    ## call the C routine
    r <- .C("blasso_R",
            T = as.integer(T),
            thin = as.integer(thin),
            cols = as.integer(m),
            n = as.integer(n),
            X = as.double(t(X)),
            y = as.double(y),
            lambda2 = as.double(rep(lambda2,T)),
            mu = double(T),
            RJ = as.integer(RJ),
            M = as.integer(M),
            beta = as.double(rep(beta, T)),
            m = as.integer(rep(sum(beta!=0), T)),
            s2 = as.double(rep(s2, T)),
            tau2i = as.double(rep(tau2i, T)),
            lpost = double(T),
            r = as.double(rd[1]),
            delta = as.double(rd[2]),
            a = as.double(ab[1]),
            b = as.double(ab[2]),
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

    ## first lpost not available
    r$lpost[1] <- NA

    ## make logicals again
    r$normalize = as.logical(r$normalize)
    r$RJ <- as.logical(r$RJ)
    r$rao.s2 <- as.logical(r$rao.s2)
    
    ## null-out redundancies
    r$n <- r$cols <- r$verb <- NULL

    ## assign call and class
    r$call <- cl
    class(r) <- "blasso"
    
    return(r)
  }


## blasso.cleanup
##
## gets called when the C-side is aborted by the R-side and enables
## the R-side to clean up the memory still allocaed to the C-side,
## as well as whatever files were left open on the C-side

"blasso.cleanup" <-  function()
{
  .C("blasso_cleanup", PACKAGE="monomvn")
}
