## regress:
##
## fit y2 ~ y1  using a linear model.
## 
## If p*nrow(y1) >= p*ncol(y1) then use principal least squares
## (pls) or principal component (pc) regression instead of lsfit

`regress` <-
function(y1, y2, method="plsr", p=1.0, ncomp.max=Inf, validation="CV", 
         verb=0, quiet=TRUE)
  {
    ## number of observatios and predictors
    numobs <- nrow(y1); if(is.null(numobs)) numobs <- length(y1)
    numpred <- ncol(y1); if(is.null(numpred)) numpred <- 1

    ## start progress meter
    if(verb > 0) cat(paste(numobs, "x", numpred, " ", sep=""))

    ## use non-LS regression when usepler-times the number of columns
    ## in the regression is >= the number of rows (length of y2)
    if(!is.null(dim(y1)) && (numpred > 1) && (numpred >= p*numobs)) {

      ## choose the regression method
      if(method == "plsr" || method == "pcr")
        ret <- regress.pls(y1, y2, method, ncomp.max, validation, verb, quiet)
      else if(method == "ridge") ret <- regress.ridge(y1, y2, verb)
      else ret <- regress.lars(y1, y2, method, validation, verb)

      ## least squares regression
    } else ret <- regress.ls(y1, y2, verb)

    ## calculate the mean-square residuals
    S <- (numobs-1)*cov(ret$res)/numobs

    ## make sure the regression was non-signuar.  If so,
    ## use force a pls (or maybe lars or ridge) regression & print a warning
    if(sum(S) == 0) {
      if(ret$method != "lsr") stop(paste("singular", method, "regression"))
      if(!quiet)
        warning(paste("singular least-squares ", nrow(y1), "x", numpred,
                    " regression, forcing pslr", sep=""))
      if(verb > 0) cat("[FAILED], ")

      ## using plsr
      return(regress(y1, y2, method, p=0, verb=verb))
    }

    ## return method, mean vector, and mean-squared of residuals
    return(list(method=ret$method, ncomp=ret$ncomp, b=ret$b, S=S))
  }

