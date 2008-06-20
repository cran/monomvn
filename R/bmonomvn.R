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


## bmonomvn:
##
## Under a MVN model, sample from the posterior distribution of the
## mean and variance matrix from a data matrix y that is potentially
## subject to monotone missingness.  The rows need not be sorted to
## give a monotone appearance, but the pattern must be monotone.

'bmonomvn' <-
function(y, pre=TRUE, p=0.9, B=100, T=200, thin=10,
         method=c("default", "rjlasso", "rjlsr", "lasso"),
         capm=method!="lasso", start=NULL, r=2, delta=0.1,
         rao.s2=TRUE, verb=1, trace=FALSE)
  {
    ## (quitely) double-check that blasso is clean before-hand
    bmonomvn.cleanup(1)
    
    ## what to do if fatally interrupted?
    on.exit(bmonomvn.cleanup(verb))
    
    ## save column names in a data frame, and then work with a matrix
    nam <- colnames(y)
    y <- as.matrix(y)
  
    ## dimensions of the inputs
    M <- ncol(y)
    N <- nrow(y)
    
    ## check p argument
    if(length(p) != 1 || p > 1 || p < 0) {
      warning("should have scalar 0 <= p <= 1, using default p=1")
      p <- 1
    }

    ## check B
    if(length(B) != 1 || B < 0)
      stop("B must be a scalar integer >= 0")
    
    ## check T
    if(length(T) != 1 || (T <= 0 && B > 0))
      stop("if B>0 when T must be a scalar integer >= 1")

    ## check thin
    if(length(thin) != 1 || thin < 1)
      stop("thin must be a scalar integer >= 1")

    ## check method
    method <- match.arg(method)

    ## change method into an integer
    mi <- 0  ## for rjlasso 
    if(method == "rjlsr") mi <- 1
    else if(method == "lasso") mi <- 2
    else if(method == "default") mi <- 3
    
    ## check capm
    if(length(capm) != 1 || !is.logical(capm))
      stop("capm must be a scalar logical or \"p\"\n")
    if(method == "lasso" && capm != FALSE) {
      warning("capm must be FALSE for method=\"lasso\"", immediate.=TRUE)
      capm <- FALSE
    }
        
    ## save the call
    cl <- match.call()
    
    ## get the number of nas in each column
    nas <- apply(y, 2, function(x) {sum(is.na(x))})

    ## check for cols with all NAs
    if(sum(nas == N) > 0) {
      cat("cols with no data:\n")
      print((1:M)[nas == N])
      stop("remove these columns and try again")
    }
    
    ## re-order the columns to follow the monotone pattern
    if(pre) {
      nao <- order(nas)
      y <- y[,nao]
    } else {
      nao <- 1:ncol(y)
    }
    
    ## check the start argument
    start <- check.start(start, nao, M)
    
    ## number of non-nas in each column
    n <- N - nas[nao]

    ## replace NAs with zeros
    Y <- y
    Y[is.na(Y)] <- 0

    if(verb >=1) cat("\n")
    
    ## call the C routine
    r <- .C("bmonomvn_R",
            B = as.integer(B),
            T = as.integer(T),
            thin = as.integer(thin),
            M = as.integer(M),
            N = as.integer(N),
            Y = as.double(t(Y)),
            n = as.integer(n),
            p = as.double(p),
            mi = as.integer(mi),
            capm = as.integer(capm),
            smu = as.double(start$mu),
            sS = as.double(start$S),
            sncomp = as.integer(start$ncomp),
            slambda = as.double(start$lambda),
            r = as.double(r),
            delta = as.double(delta),
            rao.s2 = as.integer(rao.s2), 
            verb = as.integer(verb),
            trace = as.integer(trace),
            mu = double(M),
            mu.var = double(M),
            S = double(M*M),
            S.var = double(M*M),
            methods = integer(M),
            lambda2 = double(M),
            ncomp = double(M),
            PACKAGE = "monomvn")

    ## copy the inputs back into the returned R-object
    r$Y <- NULL; r$y <- y

    ## make S into a matrix
    r$S <- matrix(r$S, ncol=M)
    r$S.var <- matrix(r$S.var, ncol=M)

    ## possibly add column permutation info from pre-processing
    if(pre) {
      r$na <- nas
      r$o <- nao
    }
    
    ## extract the methods
    mnames <- c("bcomplete", "brjlasso", "brjlsr", "blasso", "blsr")
    r$methods <- mnames[r$methods]
    
    ## put the original ordering back
    if(pre) {
      oo <- order(nao)
      r$mu <- r$mu[oo]
      r$mu.var <- r$mu.var[oo]
      r$S <- r$S[oo,oo]
      r$S.var <- r$S.var[oo,oo]
      r$ncomp <- r$ncomp[oo]
      r$lambda2 <- r$lambda2[oo]
      r$methods <- r$methods[oo]
    } else oo <- NULL

    ## deal with names
    if(! is.null(nam)) {
      r$mu <- matrix(r$mu, nrow=length(r$mu))
      rownames(r$mu) <- nam
      r$mu.var <- matrix(r$mu.var, nrow=length(r$mu.var))
      rownames(r$mu.var) <- nam
      colnames(r$S) <- rownames(r$S) <- nam
      colnames(r$S.var) <- rownames(r$S.var) <- nam
      r$ncomp <- matrix(r$ncomp, nrow=length(r$ncomp))
      rownames(r$ncomp) <- nam
      r$lambda2 <- matrix(r$lambda2, nrow=length(r$lambda2))
      rownames(r$lambda2) <- nam
    }

    ## read the trace in the output files, and then delete them
    if(trace) r$trace <- bmonomvn.read.traces(r$N, r$n, r$M, oo, nam, r$verb)
    else r$trace <- NULL
    
    ## final line
    if(verb >= 1) cat("\n")

    ## null-out redundancies
    r$n <- r$N <- r$M <- r$mi <- r$verb <- NULL
    r$smu <- r$sS <- r$sncomp <- r$slambda <- NULL

    ## change back to logicals or original inputs
    r$rao.s2 <- as.logical(r$rao.s2)
    r$capm <- as.logical(r$capm)
    
    ## assign class, call and methods, and return
    r$call <- cl
    class(r) <- "monomvn"
    return(r)
  }


## check.start:
##
## sanity check the format of the start vector, and then
## re-arrange the components into the monotone order
## specified in nao, which should agree with start$o

check.start <- function(start, nao, M)
{
  s <- list(mu=NULL, S=NULL, ncomp=NULL)
  
  if(!is.null(start)) {
    
    ## make sure orders are the smae
    if(!is.null(start$o) && start$o != nao)
      stop("starting monotone order is not the same as y's order")
    
    ## check and the reorder mu
    if(!is.null(start$mu) && length(start$mu) == M) s$mu <- start$mu[nao]
    else stop("start$mu must be specified and have length ncol(y)")
    
    ## check and then reorder S
    if(!is.null(start$S) && nrow(start$S) == ncol(start$S) && nrow(start$S) == M)
      s$S <- start$S[nao, nao]
    else stop("start$S must be specified, be square, and have dim = ncol(y)")
    
    ## check and then reorder ncomp
    if(!is.null(start$ncomp)) {
      s$ncomp <- start$ncomp
      na <- is.na(s$ncomp)
      s$ncomp <- s$ncomp[nao]
      s$ncomp[na[nao]] <- (0:(length(s$ncomp)-1))[na[nao]]
      if(length(s$ncomp) != M || !is.integer(s$ncomp) || any(s$ncomp < 0) )
        stop("start$ncomp should be a non-neg integer vector of length ncol(y)")
    } else s$ncomp <- 0:(M-1)

    ## check and then reorder lambda
    if(!is.null(start$lambda)) {
      s$lambda <- start$lambda
      na <- is.na(s$lambda)
      s$lambda <- s$lambda[nao]
      s$lambda[na[nao]] <- 0
      if(length(s$lambda) != M || any(s$lambda < 0) )
        stop("start$lambda should be a non-neg vector of length ncol(y)")
    } else s$lambda <- rep(0, M)

  }
  return(s)
}


## bmonomvn.read.traces:
##
## read the traces contained in the files written by the bmonomvn
## C-side, process them as appropriate, and then delete the trace files

"bmonomvn.read.traces" <-
  function(N, n, M, oo, nam, verb, rmfiles=TRUE)
{
  trace <- list()
  if(verb >= 1) cat("\nGathering traces\n")
  
  ## read trace of the mean samples (mu)
  if(file.exists(paste("./", "mu.trace", sep=""))) {
    trace$mu <- read.table("mu.trace")
    if(!is.null(oo)) trace$mu <- trace$mu[,oo]
    if(!is.null(nam)) names(trace$mu) <- nam
    if(rmfiles) unlink("mu.trace")
    if(verb >= 1) cat("  mu traces done\n")
  }

  ## read trace of the Covar samples (S)
  if(file.exists(paste("./", "S.trace", sep=""))) {
    trace$S <- read.table("S.trace")

    ## reorder the columns
    if(!is.null(oo)) {
      om <- matrix(NA, M, M)
      om[lower.tri(om, diag=TRUE)] <- 1:((M+1)*M/2)
      om[upper.tri(om)] <- t(om)[upper.tri(t(om))]
      om <- om[oo,oo]
      om <- om[lower.tri(om, diag=TRUE)]
      trace$S <- trace$S[,om]
    }

    ## assign names to the columns
    if(is.null(nam)) nam <- 1:length(n)
    namm <- rep(NA, (M+1)*M/2)
    k <- 1
    for(i in 1:M) for(j in i:M) {
      namm[k] <- paste(nam[i], ":", nam[j], sep="")
      k <- k+1
    }
    names(trace$S) <- namm

    ## delete the trace file
    if(rmfiles) unlink("S.trace")
    if(verb >= 1) cat("   S traces done\n")
  }

  ## read the blasso regression traces
  for(i in 1:length(n)) {
    fname <- paste("blasso_M", i-1, "_n", n[i], ".trace", sep="")
    lname <- paste("M", i-1, ".n", n[i], sep="")
    trace$reg[[lname]] <- read.table(fname, header=TRUE)
    if(rmfiles) unlink(fname)

    ## progress meter
    if(verb >= 1) {
      if(i==length(n)) cat(" reg traces 100% done  \r")
      else cat(paste(" reg traces ", round(100*i/length(n)),
                     "% done   \r", sep=""))
    }
  }

  ## cap off with a final newline
  if(verb >= 1) cat("\n")

  return(trace)
}


## bmonomvn.cleanup
##
## gets called when the C-side is aborted by the R-side and enables
## the R-side to clean up the memory still allocaed to the C-side,
## as well as whatever files were left open on the C-side

"bmonomvn.cleanup" <-  function(verb)
{
  .C("bmonomvn_cleanup", PACKAGE="monomvn")

  ## get rid of trace of the mean samples (mu)
  if(file.exists(paste("./", "mu.trace", sep=""))) {
    unlink("mu.trace")
    if(verb >= 1) cat("NOTICE: removed mu.trace\n")
  }

  ## get rid of trace of the Covar samples (S)
  if(file.exists(paste("./", "S.trace", sep=""))) {
    unlink("S.trace")
    if(verb >= 1) cat("NOTICE: removed S.trace\n")
  }

  ## get all of the names of the tree files
  b.files <- list.files(pattern="blasso_M[0-9]+_n[0-9]+.trace")
    
  ## for each tree file
  if(length(b.files > 0)) {
    for(i in 1:length(b.files)) {
      if(verb >= 1) cat(paste("NOTICE: removed ", b.files[i], "\n", sep=""))
      unlink(b.files[i])
    }
  }
}
