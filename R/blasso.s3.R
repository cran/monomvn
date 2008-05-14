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


## plot.blasso
##
## generic plotting method for blasso class objects.
## plots summaries of the samples from
## from the posterior distribution of the Bayesian lasso
## model

'plot.blasso' <-
  function(x, which=c("coef", "s2", "lambda2", "tau2i"), burnin=0, ...)
{
  ## check the burnin argument
  if(length(burnin) != 1 || burnin<0 || burnin >= x$T)
    stop("burnin must be a non-negative scalar < x$T")
  
  ## check the which argument
  which <- match.arg(which)

  ## make the appropriate kind of plot
  if(which == "coef") {
    boxplot(data.frame(mu=x$mu[burnin:x$T], x$beta[burnin:x$T,]),
            ylab="coef", main="Boxplots of regression coefficients", ...)
  } else if(which == "s2") hist(x$s2[burnin:x$T], ...)
  else if(which == "lambda2") hist(x$lambda2[burnin:x$T], ...)
  else if(which == "lambda2"){
    boxplot(data.frame(x$tau2i[burnin:x$T,]),
            main="Boxplot of tau2i", ylab="tau2i", ...)
  }
}


## summary.blasso
##
## generic summary method for blasso class objects,
## basically calls summary on the matrices and vectos
## of samples from the posterior distribution of the
## parameters in the Bayesian lasso model

'summary.blasso' <-
function(object, burnin=0, ...)
  {

    ## check the burnin argument
    if(length(burnin) != 1 || burnin<0 || burnin >= object$T)
      stop("burnin must be a non-negative scalar < object$T")

    ## make the list
    rl <- list(call=object$call, B=burnin, T=object$T, thin=object$thin)
    class(rl) <- "summary.blasso"
    
    ## call summary on each object
    df <- data.frame(mu=object$mu[burnin:object$T],
                     object$beta[burnin:object$T,])
    rl$coef <- summary(df)
    rl$s2 <- summary(object$s2[burnin:object$T])
    rl$lambda2 <- summary(object$lambda2[burnin:object$T])
    rl$tau2i <- summary(data.frame(object$tau2i[burnin:object$T,]))

    ## print it or return it
    rl
  }


## print.summary.blasso
##
## print the results of the summary method after first
## calling the print method on the blasso object.  

'print.summary.blasso' <-
  function(x, ...)
{
  ## print information about the call
  cat("\nCall:\n")
  print(x$call)
  
  ## print the monomvn object
  cat("\nsummary of MCMC samples with B=", x$B, " burnin rounds\n", sep="")
  cat("T=", x$T, " total rounds, with thin=", x$thin,
      " rounds between\n", sep="")
  cat("each sample\n\n")
  
  ## print coef
  cat("coefficients:\n")
  print(x$coef)
  cat("\n")

  ## print s2
  cat("s2:\n")
  print(x$s2)
  cat("\n")

  ## print lambda2
  cat("lambda2:\n")
  print(x$lambda2)
  cat("\n")

  ## print tau2i
  cat("tau2i:\n")
  print(x$tau2i)
  cat("\n")
}


## print.blasso
##
## generic print method for blasso class objects,
## summarizing the results of a blasso call

`print.blasso` <-
function(x, ...)
  {
    ## print information about the call
    cat("\nCall:\n")
    print(x$call)

    ## print the monomvn object
    cat("\nrun for T=", x$T, " MCMC samples, with thin=", x$thin,
        " rounds between\n", sep="")
    cat("each sample\n")

    ## suggestion
    cat("\nTry summary.blasso and plot.blasso on this object\n\n")
  }
