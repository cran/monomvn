## regress.ridge:
##
## fit y2 ~ y1  using ridge regression
##
## NOTE: this does not seem to work for the regression with
## "big-p small-n" -- it essentially gives zero-residulas
## regardless of the method used to choose the ridge
## constant (lambda)


opt.ridge <- function(lambda, y2, y1)
  {
    if(lambda <= 0) return(Inf)
        return(lm.ridge(y2 ~ y1, lambda=lambda)$GCV)
  }

'regress.ridge' <-
  function(y1, y2, verb=0)
{
  ## number of regressions
  numreg <- ncol(y2); if(is.null(numreg)) numreg <- 1
    
  ## number of observatios and predictors
  numobs <- nrow(y1); if(is.null(numobs)) numobs <- length(y1)
  numpred <- ncol(y1); if(is.null(numpred)) numpred <- 1

  ## add to progress meter
  method <- "ridge"
  if(verb > 0) cat(paste("using ", method, " (LOO) ", sep=""))

  ## choose lambda (could be different for each response)
  lambda <- rep(NA, numreg)
  bvec <- matrix(NA, nrow=numpred+1, ncol=numreg)
  res <- matrix(NA, nrow=numobs, ncol=numreg)
  if(verb > 0) cat("lambda:")

  ## a hueristic for lower-bounding and upper-bounding of lambda
  lmin <- 0
  if(numpred > numobs) lmin <- log(numpred-numobs)
  lower <- lmin; upper <- numpred
  
  if(numreg == 1) y2 <- matrix(y2, ncol=numreg)
  for(i in 1:numreg) {
    
    ## actually get the ridge reggression constant (lambda)
    while(1) {
      lambda <- optimize(opt.ridge, lower=lower, upper=upper,
                         y2=y2[,i], y1=y1)$minimum
      if(round(lambda) != upper) break;
      lower <- lambda; upper <- 2*upper
    }

    ## use that lambda, and re-fit the model to get the coefficients
    reglst <- lm.ridge(y2[,i] ~ y1, lambda=lambda)
    bvec[,i] <- coef(reglst)
    lambda[i] <- reglst$lambda
    if(verb > 0) cat(paste(" ", signif(lambda[i],5), sep=""))
    
    ## use predict to get the residuals
    ## res[,i] <- y2[,i] - predict(reglst, s=s, newx=y1, mode="fraction")$fit
    res[,i] <- y2[,i] - (bvec[1,i] + y1 %*% bvec[-1,i])
  }
  if(verb > 0) cat(paste(" in range [", signif(lmin,5),
                         ",", upper, "]", sep=""))

  return(list(method=method, ncomp=lambda, b=bvec, res=res))
}
