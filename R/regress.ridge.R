## regress.ridge:
##
## fit y2 ~ y1  using ridge regression
##
## NOTE: this does not seem to work for the regression with
## "big-p small-n" -- it essentially gives zero-residulas
## regardless of the method used to choose the ridge
## constant (lambda)


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
  if(verb > 0) cat(paste("using ", method, " ", sep=""))

  ## choose lambda (could be different for each response)
  lambda <- rep(NA, numreg)
  bvec <- matrix(NA, nrow=numpred+1, ncol=numreg)
  res <- matrix(NA, nrow=numobs, ncol=numreg)
  if(verb > 0) cat("lambda:")
  
  if(numreg == 1) y2 <- matrix(y2, ncol=numreg)
  for(i in 1:numreg) {
    
    ## actually get the ridge reggression constant (lambda) 
    r1 <- lm.ridge(y2[,i] ~ y1)
    reglst <- lm.ridge(y2[,i] ~ y1, lambda=r1$kLW)#kHKB)
    bvec[,i] <- coef(reglst)
    lambda[i] <- reglst$lambda
    if(verb > 0) cat(paste(" ", signif(lambda[i],5), sep=""))
    
    ## use predict to get the residuals
    ## res[,i] <- y2[,i] - predict(reglst, s=s, newx=y1, mode="fraction")$fit
    res[,i] <- y2[,i] - (bvec[1,i] + y1 %*% bvec[-1,i])
    print(res[,i])
  }
  ##if(verb > 0) cat(paste(" of ", min(numobs, numpred), sep=""))

  return(list(method=method, ncomp=lambda, b=bvec, res=res))
}
