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


## bridge:
##
## Bayesian Ridge regress.  Simply calls the blasso function with
## the ridge argument modified to be TRUE and the rd argment of c(0,0)
## for a Jeffrey's prior on the squared ridge penalty lambda2

'bridge' <-
function(X, y, T=1000, thin=NULL, RJ=TRUE, M=NULL, beta=NULL,
         lambda2=1, s2=1, mprior=0, rd=NULL, ab=NULL,
         rao.s2=TRUE, normalize=TRUE, verb=1)
  {
    blasso(X=X, y=y, T=T, thin=thin, RJ=RJ, M=M, beta=beta,
           lambda2=lambda2, s2=s2, ridge=TRUE, mprior=mprior,
           rd=rd, ab=ab, rao.s2=rao.s2, normalize=normalize,
           verb=verb)
  }


## blasso:
##
## function for sampling from the posterior distribution of the
## regression coefficients (beta) and variance (s2) under the
## Bayesian Lasso linear model of Park & Casella

'blasso' <-
function(X, y, T=1000, thin=NULL, RJ=TRUE, M=NULL, beta=NULL,
         lambda2=1, s2=1, ridge=FALSE, mprior=0, rd=NULL,
         ab=NULL, rao.s2=TRUE, normalize=TRUE, verb=1)
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

    ## check RJ (reversible jump)
    if(length(RJ) != 1 || !is.logical(RJ))
      stop("RJ must be a scalar logical\n")

    ## check lambda2
    if(length(lambda2) != 1 || lambda2 < 0)
      stop("lambda2 must be a non-negative scalar")

    ## check ridge
    if(length(ridge) != 1 || !is.logical(ridge))
      stop("ridge should be a logical scalar")
    if(ridge && lambda2 == 0)
      stop("specifying ridge=TRUE and lambda2=0 doesn't make any sense")
    
    ## check M or default
    if(is.null(M)) {
      if(RJ) M <- as.integer(min(m, n-1))
      else M <- m
    }
    M <- as.integer(M)
    if(length(M) != 1 || M < 0 || M > m)
      stop("M must be a positive integer 0 <= M <= ncol(X)")
    if(!RJ && M != m) {
      M <- m
      warning("must have M=", M, " == ncol(X)=", m, " when RJ=FALSE",
              immediate.=TRUE)
    }
    
    ## check beta or default
    if(is.null(beta)) beta <- rep(!RJ, m)
    if(length(beta) != m)
      stop("must have length(beta) == ncol(X)")

    ## general big-p small-n problem -- must have lasso on
    if(lambda2 == 0 && m >= n)
      stop("big p small n problem; must have lambda2 > 0")
    else if(lambda2 == 0) lambda2 <- double(0)
    else lambda2 <- as.double(rep(lambda2, T))
    
    ## check for a valid regression
    if(!RJ) {  ## when not doing reversible jump (RJ)
      if(length(lambda2) > 0 && any(beta == 0)) {
        warning("must start with non-zero beta when RJ=FALSE, using beta=1\n",
                immediate.=TRUE)
        beta  <- rep(!RJ, m)
      }
    } else if(sum(beta != 0) > M) { ## when doing RJ
      beta <- rep(0, m)
      warning("initial beta must have M or fewer non-zero entries",
              immediate.=TRUE)
    }

    ## check thin or default
    if(is.null(thin)) {
      if(RJ || (length(lambda2) > 0 && !ridge)) thin <- M
      else if(length(lambda2) > 0) thin <- 2
      else thin <- 1
    }
    if(length(thin) != 1 || thin < 1)
      stop("thin must be a scalar integer >= 1")

    ## check s2
    if(length(s2) != 1 || s2 <= 0)
      stop("s2 must be a positive scalar")

    ## check tau2i or default
    if(ridge || length(lambda2) == 0) tau2i <- double(0)
    else {
      tau2i <- rep(1, m)
      tau2i[beta == 0] <- -1
      tau2i <- as.double(rep(tau2i, T))
    }

    ## check mprior
    if(length(mprior) != 1 || mprior < 0 || mprior > 1) {
      stop("mprior should be a scalar 0 <= mprior < 1");
    } else if(mprior != 0 && RJ == FALSE) {
      warning(paste("setting mprior=", mprior, " ignored since RJ=FALSE",
                    sep=""))
    }

    ## check r and delta (rd)
    if(is.null(rd)) {
      if(ridge) { ## if using ridge regression IG prior
        rd <- c(0,0)
        ## big-p small-n setting for ridge
        if(m >= n) rd <- c(5, 10) 
      } else rd <- c(2, 0.1) ## otherwise lasso G prior
    }
    ## double-check rd
    if(length(rd) != 2 || (length(tau2i) > 0 && any(rd <= 0)))
      stop("rd must be a positive 2-vector")

    ## check ab or default
    if(is.null(ab)) {
      ab <- c(0,0)
      if(!RJ && lambda2 > 0 && m >= n) {
        ab[1] <- 3/2
        ab[2] <- Igamma.inv(ab[1], 0.95*gamma(ab[1]), lower=FALSE)*sum(y^2)
      } 
    }

    ## double check ab
    if(length(ab) != 2 || any(ab < 0))
      stop("ab must be a non-negative 2-vector")
    if(!ridge && !RJ && m >= n && any(ab <= 0))
      stop("must have ab > c(0,0) when !ridge, !RJ, and ncol(X) >= length(y)")

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
            lambda2 = lambda2,
            mu = double(T),
            RJ = as.integer(RJ),
            M = as.integer(M),
            beta = as.double(rep(beta, T)),
            m = as.integer(rep(sum(beta!=0), T)),
            s2 = as.double(rep(s2, T)),
            tau2i = tau2i,
            lpost = double(T),
            mprior = as.double(mprior),
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

    if(r$lambda[1] != 0 && length(r$tau2i) > 0) {
      r$tau2i <- matrix(r$tau2i, nrow=T, ncol=m, byrow=TRUE,
                        dimnames=list(NULL,paste("tau2i.", 1:m, sep="")))

      ## put NAs where tau2i has -1
      r$tau2i[r$tau2i == -1] <- NA
    } else if(length(r$tau2i) > 0) {
      r$lambda <- r$tau2i <- NULL 
    } else r$tau2i <- NULL
    
    ## first lpost not available
    r$lpost[1] <- NA

    ## make logicals again
    r$normalize = as.logical(r$normalize)
    r$RJ <- as.logical(r$RJ)
    r$rao.s2 <- as.logical(r$rao.s2)

    ## null-out redundancies
    r$col <- r$n <- r$cols <- r$verb <- NULL
    if(length(r$lambda2) == 0) r$lambda2 <- NULL

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
