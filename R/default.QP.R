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


## check.QP:
##
## check the Quadratic Programming arguments, as described
## by solve.QP in library(quadprog) as encoded in QP
## and reorder by nao

check.QP <- function(QP, m, nao)
  {
    ## check if we should return the null default solve.QP params
    if(is.null(QP) || (is.logical(QP) && QP == FALSE))
      return(list(dvec=NULL, dmu=FALSE, Amat=NULL, b0=NULL,
                  mu.constr=0, q=0, meq=0))
    
    ## use the default.QP, and then continue to check
    if(is.logical(QP) && QP == TRUE) QP <- default.QP(m)
    
    ## make sure QP is a list
    if(!is.list(QP)) stop("QP should be a list object")

    ## check the Amat argument
    Amat <- as.matrix(QP$Amat)
    if(nrow(Amat) != m) stop("QP$Amat should have", m, " rows")
    if(!is.null(nao)) Amat <- Amat[nao,]

    ## check the b0 argument
    b0 <- as.vector(QP$b0)
    if(length(b0) != ncol(Amat))
      stop("should have length(QP$b0) == ncol(QP$Amat)")

    ## check the meq argument
    meq <- as.integer(QP$meq)
    if(length(meq) != 1 || meq > ncol(Amat) || meq < 0)
      stop("must have 0 <= QP$meq <= ncol(QP$Amat) =", ncol(Amat))
    
    ## check the dvec argument
    dvec <- as.vector(QP$dvec)
    if(is.null(dvec) || length(dvec) != m)
        stop("QP$dvec) must be a vector of length", m)

    ## check the dmu argument
    dmu <- QP$dmu
    if(is.null(dmu) || !is.logical(dmu) || length(dmu) != 1)
      stop("QP$dmu must be a scalar logical")

    ## check the mu.constr argument
    mu.constr <- QP$mu.constr
    if(is.null(mu.constr) || !is.numeric(mu.constr) ||
       any(mu.constr[-1] < 1) || length(mu.constr)-1 != mu.constr[1] ||
       any(duplicated(mu.constr[-1])) )
      stop("QP$mu.costr must be a positive integer vector with\n",
           "\tlength(mu.constr)-1 = mu.constr[1] and no duplicated\n",
           "\tentries in mu.constr[-1]")

    ## return the list
    return(list(dvec=dvec, dmu=dmu, Amat=Amat, b0=b0,
                mu.constr=mu.constr, q=ncol(Amat), meq=meq))
  }


## default.QP:
##
## create the devault solve.QP setup that minimizes
## the variance

default.QP <- function(m, dmu=FALSE, mu.constr=NULL)
  {
    ## the sum of weights must be equal to 1
    Amat <- matrix(rep(1, m), ncol=1)
    b0 <- 1
    meq <- 1

    ## each w one must be positive
    Amat <- cbind(Amat, diag(rep(1, m)))
    b0 <- c(b0, rep(0, m))

    ## each one less than 1
    Amat <- cbind(Amat, diag(rep(-1, m)))
    b0 <- c(b0, rep(-1, m))

    ## construct the dvec and dmu
    if(!is.logical(dmu) || length(dmu) != 1)
      stop("dmu should be a scalar logical")
    if(dmu) dvec <- rep(1, m)
    else dvec <- rep(0, m)

    ## check for a constraint on mu
    if(!is.null(mu.constr)) {
      if(!is.numeric(mu.constr))
        stop("mu.constr should numeric or NULL")

      ## create columns to add on to Amat that are ones that
      ## alternate in sign 
      addc <- matrix(1, ncol=length(mu.constr), nrow=m)
      parity <- rep(c(1,-1), ceiling(ncol(addc)/2))
      if(ceiling(ncol(addc)/2) != floor(ncol(addc)/2))
        parity <- parity[-length(parity)]
      parity <- matrix(parity, nrow=nrow(Amat), ncol=length(parity), byrow=TRUE)
      addc <- addc * parity

      ## add those columns to Amat, and the constrants to b0
      Amat <- cbind(Amat, addc)
      b0 <- c(b0, mu.constr)

      ## record the length and which colums were added
      mu.constr <- c(length(mu.constr),
                     (ncol(Amat)-(length(mu.constr)-1)):ncol(Amat))
    } else mu.constr <- 0

    ## return the list
    return(list(dvec=dvec, dmu=dmu, Amat=Amat, b0=b0,
                mu.constr=mu.constr, meq=meq))
  }
