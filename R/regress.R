## regress:
##
## fit y2 ~ y1  using a linear model.
## 
## If p*nrow(y1) >= p*ncol(y1) then use principal least squares
## (pls) or principal component (pc) regression instead of lsfit

`regress` <-
function(y1, y2, method="plsr", p=1.0, ncomp.max=Inf, verb=0, quiet=TRUE)
  {
    ## number of regressions
    numreg <- ncol(y2); if(is.null(numreg)) numreg <- 1
    
    ## number of observatios and predictors
    numobs <- nrow(y1); if(is.null(numobs)) numobs <- length(y1)
    numpred <- ncol(y1); if(is.null(numpred)) numpred <- 1

    ## start progress meter
    if(verb > 0) cat(paste(numobs, "x", numpred, " ", sep=""))

    ## use pls/pc regression when usepler-times the number of columns
    ## in the regression is >= the number of rows (length of y2)
    if(!is.null(dim(y1)) && (numpred > 1) && (numpred >= p*numobs)) {

      ## force plsr when there are rediculiously few data points
      if(numobs <= 3 && method == "pcr") {
        if(!quiet) warning("forcing plsr when fewer than 4 non-missing entries")
        method <- "plsr"
      }
      
      ## decide on cross validation method depending on number of data points
      validation <- "CV"
      if(numobs <= 10) validation <- "LOO"
      actual.method <- paste(method, "-", validation, sep="")
      
      ## add to progress meter
      if(verb > 0) cat(paste("using ", method, " (", validation, ") ", sep=""))

      ## maximum number of components to include in the regression
      ncomp.max <- min(ncomp.max, numpred, numobs-1)

      ## use principal least squares or principal component regression
      suppressWarnings({
      if(method == "plsr") reglst <- plsr(y2~y1,validation=validation,ncomp=ncomp.max)
      else if(method == "pcr") reglst <- pcr(y2~y1,validation=validation,ncomp=ncomp.max)
      else stop(paste("regression method", method, "unknown"))
      })
                       
      ## choose number of components (could be different for each response)
      ncomp <- rep(NA, numreg)
      bvec <- matrix(NA, nrow=numpred+1, ncol=numreg)
      res <- matrix(NA, nrow=numobs, ncol=numreg)
      a <- matrix(RMSEP(reglst, estimate="CV", intercept=FALSE)$val, nrow=numreg)
      if(verb > 0) cat("ncomp:")
      for(i in 1:numreg) {
        ncomp[i] <- which.min(a[i,] + (max(a[i,])-min(a[i,]))*seq(0,1,length=ncol(a)))
        if(verb > 0) cat(paste(" ", ncomp[i], sep=""))
        bvec[,i] <- matrix(coef(reglst, ncomp=ncomp[i], intercept=TRUE), ncol=numreg)[,i]
        res[,i] <- reglst$resid[,i,ncomp[i]]
      }
      if(verb > 0) cat(paste(" of ", ncomp.max, sep=""))
      
    } else {
      
      ## add to progress meter
      if(verb > 0) cat(paste("using lsr ", sep=""))

      ## standard least-squares regression
      reglst <- lm(y2 ~ y1)
      bvec <- matrix(reglst$coef, ncol=numreg)
      res <- matrix(reglst$resid, ncol=numreg)
      actual.method <- "lsr"
      ncomp <- rep(NA, numreg)
    }

    ## calculate the mean-square residuals
    S <- (numobs-1)*cov(res)/numobs

    ## make sure the regression was non-signuar.  If so,
    ## use force a plsr regression and print a warning
    if(sum(S) == 0) {
      if(actual.method != "lsr") stop(paste("singular", method, "regression"))
      if(!quiet)
        warning(paste("singular least-squares ", nrow(y1), "x", numpred,
                    " regression, forcing pslr", sep=""))
      if(verb > 0) cat("[FAILED], ")

      ## using plsr
      return(regress(y1, y2, method, p=0, verb=verb))
    }

    ## return method, mean vector, and mean-squared of residuals
    return(list(method=actual.method, ncomp=ncomp, b=bvec, S=S))
  }

