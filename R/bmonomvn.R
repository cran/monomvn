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
function(y, pre=TRUE, p=0.9, B=100, T=200, thin=1, economy=FALSE,
         method=c("lasso", "ridge", "lsr"), RJ=c("bpsn", "p", "none"),
         capm=method!="lasso", start=NULL, mprior= 0, rd=NULL,
         rao.s2=TRUE, QP=NULL, verb=1, trace=FALSE)
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

    ## check rao.s2
    if(length(rao.s2) != 1 || !is.logical(rao.s2))
      stop("rao.s2 must be a scalar logical")

    ## check economy
    if(length(economy) != 1 || !is.logical(economy))
      stop("economy must be a scalar logical")

    ## check method, and change to an integer
    method <- match.arg(method)
    mi <- 0  ## lasso
    if(method == "ridge") mi <- 1
    else if(method == "lsr") mi <- 2

    ## check RJ, and change to an integer
    RJ <- match.arg(RJ)
    RJi <- 0  ## bpsn: "big-p small-n"
    if(RJ == "p") RJi <- 1 ## only do RJ when parsimonious is activated
    else if(RJ == "none") RJi <- 2 ## no RJ

    ## disallow bad RJ combination
    if(method == "lsr" && RJ == "none")
      stop("bad method (", method, ") and RJ (", RJ, ") combination", sep="")
    
    ## check capm
    if(length(capm) != 1 || !is.logical(capm))
      stop("capm must be a scalar logical or \"p\"\n")
    if(method == "lasso" && capm != FALSE) {
      warning("capm must be FALSE for method=\"lasso\"", immediate.=TRUE)
      capm <- FALSE
    }

    ## check mprior
    if(length(mprior) != 1 || mprior < 0 || mprior > 1) {
      stop("mprior should be a scalar 0 <= mprior < 1");
    } else if(mprior != 0 && RJ == FALSE) {
      warning(paste("setting mprior=", mprior, " ignored since RJ=FALSE",
                    sep=""))
    }

    ## check r and delta (rd), or default
    if(is.null(rd)) {
      if(method == "lasso") rd <- c(2,0.1)
      else if(method == "ridge") rd <- c(5, 10)
      else rd <- c(0,0)
    }
    if(length(rd) != 2 || (method=="lasso" && any(rd <= 0)))
      stop("rd must be a positive 2-vector")
        
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
    } else nao <- 1:ncol(y)

    ## re-order the rows by NA too, since the more general monotone
    ## pattern finding is not currently implemented
    r <- apply(y, 1, function(x){ sum(is.na(x)) })
    y <- y[order(r),nao]
    ## this could be problematic in the future if we want to take
    ## use a non--iid approach, say, with change-points on time
    ## or exponential decay
    
    ## get the R matrix and n
    R <- getR(y)
    n <- N - apply(R, 2, function(x){ sum(x == 1) })
    if(sum(R == 2) != 0) stop("missingness pattern in y is not monotone")
    ## the real n would be reduced by the number of 2's in each column
    
    ## check the start argument
    start <- check.start(start, nao, M)

    ## check the QP argument
    QPin <- check.QP(QP, M, nao)
    if(is.logical(QP) && QP == FALSE) QP <- NULL

    ## save old y and then replace NA with Inf
    Y <- y
    Y[is.na(Y)] <- 0

    if(verb >=1) cat("\n")
    
    ## call the C routine
    r <- .C("bmonomvn_R",

            ## begin estimation inputs
            B = as.integer(B),
            T = as.integer(T),
            thin = as.integer(thin),
            M = as.integer(M),
            N = as.integer(N),
            Y = as.double(t(Y)),
            n = as.integer(n),
            p = as.double(p),
            mi = as.integer(mi),
            RJi = as.integer(RJi),
            capm = as.integer(capm),
            smu = as.double(start$mu),
            sS = as.double(start$S),
            sncomp = as.integer(start$ncomp),
            slambda = as.double(start$lambda),
            mprior = as.double(mprior),
            rd = as.double(rd),
            rao.s2 = as.integer(rao.s2),
            economy = as.integer(economy),
            verb = as.integer(verb),
            trace = as.integer(trace),

            ## begin Quadratic Progamming inputs
            QPd = as.double(QPin$dvec),
            QPdmu = as.integer(QPin$dmu),
            QPA = as.double(QPin$Amat),
            QPb0 = as.double(QPin$b0),
            QPmc = as.integer(QPin$mu.constr),
            QPq = as.integer(QPin$q),
            QPmeq = as.integer(QPin$meq),
            
            ## begin estimation outputs
            mu = double(M),
            mu.var = double(M),
            S = double(M*M),
            S.var = double(M*M),
            mu.map = double(M),
            S.map = double(M*M),
            lpost.map = double(1),
            which.map = integer(1),
            methods = integer(M),
            thin.act = integer(M),
            lambda2 = double(M),
            ncomp = double(M),

            ## begin Quadratic Programming outputs
            W = double(T*M*(!is.null(QPin$Amat))),
            
            PACKAGE = "monomvn")

    ## copy the inputs back into the returned R-object
    r$Y <- NULL; r$y <- y

    ## make S into a matrix
    r$S <- matrix(r$S, ncol=M)
    r$S.var <- matrix(r$S.var, ncol=M)
    r$S.map <- matrix(r$S.map, ncol=M)

    ## possibly add column permutation info from pre-processing
    if(pre) {
      r$na <- nas
      r$o <- nao
    }
    
    ## extract the methods
    mnames <- c("bcomplete", "brjlasso", "brjridge", "brjlsr",
                "blasso", "bridge", "blsr")
    r$methods <- mnames[r$methods]
    
    ## put the original ordering back
    if(pre) {
      oo <- order(nao)
      r$mu <- r$mu[oo]
      r$mu.var <- r$mu.var[oo]
      r$mu.map <- r$mu.map[oo]
      r$S <- r$S[oo,oo]
      r$S.var <- r$S.var[oo,oo]
      r$S.map <- r$S.map[oo,oo]
      r$ncomp <- r$ncomp[oo]
      r$lambda2 <- r$lambda2[oo]
      r$methods <- r$methods[oo]
      r$thin <- r$thin.act[oo]
    } else oo <- NULL

    ## deal with names
    if(! is.null(nam)) {
      r$mu <- matrix(r$mu, nrow=length(r$mu))
      rownames(r$mu) <- nam
      r$mu.var <- matrix(r$mu.var, nrow=length(r$mu.var))
      rownames(r$mu.var) <- nam
      r$mu.map <- matrix(r$mu.map, nrow=length(r$mu.map))
      rownames(r$mu.map) <- nam
      colnames(r$S) <- rownames(r$S) <- nam
      colnames(r$S.var) <- rownames(r$S.var) <- nam
      colnames(r$S.map) <- rownames(r$S.map) <- nam
      r$ncomp <- matrix(r$ncomp, nrow=length(r$ncomp))
      rownames(r$ncomp) <- nam
      r$lambda2 <- matrix(r$lambda2, nrow=length(r$lambda2))
      rownames(r$lambda2) <- nam
    }

    ## read the trace in the output files, and then delete them
    if(trace)
      r$trace <- bmonomvn.read.traces(r$N, r$n, r$M, oo, nam, capm, mprior,
                                      cl, thin, r$verb)
    else r$trace <- NULL
    
    ## final line
    if(verb >= 1) cat("\n")

    ## null-out redundancies
    r$n <- r$N <- r$M <- r$mi <- r$verb <- NULL
    r$smu <- r$sS <- r$sncomp <- r$slambda <- NULL
    r$thin.act <- NULL

    ## change back to logicals or original inputs
    r$rao.s2 <- as.logical(r$rao.s2)
    r$capm <- as.logical(r$capm)
    r$economy <- as.logical(r$economy)
    r$RJi <- NULL; r$RJ <- RJ

    ## off-by-one
    r$which.map <- r$which.map + 1

    ## record Quadratic Programming info
    r$QPd <- r$QPA <- r$QPb0 <- r$QPq <- r$QPmeq <- NULL
    if(!is.null(QP)) {

      ## save the QP inputs
      r$QP <- QPin
      r$QP$q <- NULL

      ## convert the W-vector into a matrix
      r$W <- matrix(r$W, ncol=M, nrow=T, byrow=TRUE)
      if(!is.null(oo)) { ## reorder rows or columns
        r$QP$Amat <- r$QP$Amat[oo,]
        r$W <- r$W[,oo]
      }
      if(! is.null(nam)) { ## name rows or columns
        rownames(r$QP$Amat) <- nam
        colnames(r$W) <- nam
      }
    }
    
    ## assign class, call and methods, and return
    r$call <- cl
    class(r) <- "monomvn"
    return(r)
  }


## getR:
##
## calculate which need to be imputed (2) with DA, and
## which are simply missing in the monotone pattern (1);
## else (0)

getR <- function(y)
  {
    R <- matrix(2*as.numeric(is.na(y)), ncol=ncol(y))
    for(j in ncol(y):1) {
      for(i in nrow(y):1) {
        if(R[i,j]) R[i,j] <- 1
        else break
      }
    }
    return(R)
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
    if(!is.null(start$S) && nrow(start$S) == ncol(start$S) &&
       nrow(start$S) == M)
      s$S <- start$S[nao, nao]
    else stop("start$S must be specified, be square, and have dim = ncol(y)")
    
    ## check and then reorder ncomp
    if(!is.null(start$ncomp)) {
      s$ncomp <- start$ncomp
      na <- is.na(s$ncomp)
      s$ncomp <- s$ncomp[nao]
      s$ncomp[na[nao]] <- (0:(length(s$ncomp)-1))[na[nao]]
      if(length(s$ncomp) != M || !is.integer(s$ncomp) || any(s$ncomp < 0) )
        stop("start$ncomp must be non-neg integer vector of length ncol(y)")
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
  function(N, n, M, oo, nam, capm, mprior, cl, thin, verb, rmfiles=TRUE)
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
    table <- read.table(fname, header=TRUE)
    trace$reg[[lname]] <- table2blasso(table, thin, mprior, capm,
                                       i-1, n[i], cl)
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


## table2blasso:
##
## change the table trace read in and convert it into a
## skeleton blasso class object so that the blasso methods
## like print, plot, and summary can be used

table2blasso <- function(table, thin, mprior, capm, m, n, cl)
  {
    ## first convert to a list
    tl <- as.list(table)
    
    ## start with the easy scalars
    l <- list(lpost=tl[["lpost"]], s2=tl[["s2"]], mu=tl[["mu"]],
              m=tl[["m"]], lambda2=tl[["lambda2"]])

    ## now the vectors
    bi <- grep("beta.[0-9]+", names(table))
    l$beta <- as.matrix(table[,bi])
    ti <- grep("tau2i.[0-9]+", names(table))
    l$tau2i <- as.matrix(table[,ti])
    l$tau2i[l$tau2i == -1] <- NA

    ## assign "inputs"
    l$T <- nrow(l$beta)
    l$thin <- "dynamic"
    l$RJ <- !is.null(l$m)
    l$mprior <- mprior
    if(capm) l$M <- max(m, n) 
    else l$M <- m
    
    ## assign the call and the class
    l$call <- cl
    class(l) <- "blasso"
    
    return(l)
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
  nl <- FALSE
  if(file.exists(paste("./", "S.trace", sep=""))) {
    unlink("S.trace")
    if(verb >= 1) cat("NOTICE: removed S.trace\n")
    nl <- TRUE
  }

  ## get all of the names of the tree files
  b.files <- list.files(pattern="blasso_M[0-9]+_n[0-9]+.trace")
    
  ## for each tree file
  if(length(b.files > 0)) {
    for(i in 1:length(b.files)) {
      if(verb >= 1) cat(paste("NOTICE: removed ", b.files[i], "\n", sep=""))
      unlink(b.files[i])
      nl <- TRUE
    }    
  }

  ## final newline
  if(verb >= 1 && nl) cat("\n")
}
