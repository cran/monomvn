## regress.lars:
##
## fit y2 ~ y1  using lasso.


'regress.lars' <-
  function(y1, y2, method="lasso", validation="CV", verb=0)
{
  ## number of regressions
  numreg <- ncol(y2); if(is.null(numreg)) numreg <- 1
    
  ## number of observatios and predictors
  numobs <- nrow(y1); if(is.null(numobs)) numobs <- length(y1)
  numpred <- ncol(y1); if(is.null(numpred)) numpred <- 1

  ## decide on cross validation method depending on number of data points
  if(numobs <= 10) validation <- "LOO";
  if(validation == "LOO") K <- numobs
  else K <- 10
  actual.method <- paste(method, "-", validation, sep="")

  ## add to progress meter
  if(verb > 0) cat(paste("using ", method, " (", validation, ") ", sep=""))

  ## choose lambda (could be different for each response)
  nzb<- rep(NA, numreg)
  bvec <- matrix(NA, nrow=numpred+1, ncol=numreg)
  res <- matrix(NA, nrow=numobs, ncol=numreg)
  if(verb > 0) cat("ncomp:")

  ## don't use gram matrix when m > 500
  use.Gram <- TRUE
  if(numpred > 500) use.Gram <- FALSE
  
  if(numreg == 1) y2 <- matrix(y2, ncol=numreg)
  for(i in 1:numreg) {
    
    ## using "one-standard error rule"
    ## num.fractions could be passed in same as ncomp.max
    cv <- cv.lars(x=y1,y=y2[,i],type=method,K=K,intercept=TRUE,
                  plot.it=FALSE, use.Gram=use.Gram)
    wm <- which.min(cv$cv)
    tf <- cv$cv < cv$cv[wm] + cv$cv.error[wm]
    s <- (1:100)[tf][1]/100
    ## abline(v=s)

    ## first imterpretation of one-standard error rule
    ## tf <- cv$cv - cv$cv.error < min(cv$cv)
    ## s <- (1:100)[tf][1]/100
    ## abline(v=s, col=2)

    ## the simple line heuristic
    ## s <- (1:100)[which.min(cv$cv + (max(cv$cv)-min(cv$cv))*seq(0,1,length=length(cv$cv)))][1]/100
    ## abline(v=s, col=3, lty=3);

    ## wait for next plot
    ## readline("pr: ")
    
    ## get the lasso fit with fraction f
    reglst <- lars(x=y1,y=y2[,i],type=method,intercept=TRUE, use.Gram=use.Gram)
    co <- coef(reglst, s=s, mode="fraction")
    y1co <- drop(y1 %*% co)
    icept <- reglst$mu - mean(y1co)
    bvec[,i] <- c(icept, co)
    nzb[i] <- sum(co != 0)
    if(verb > 0) cat(paste(" ", nzb[i], sep=""))
    
    ## use predict to get the residuals
    ## res[,i] <- y2[,i] - predict(reglst, s=s, newx=y1, mode="fraction")$fit
    res[,i] <- y2[,i] - (icept + y1co)
  }
  if(verb > 0) cat(paste(" of ", min(numobs, numpred), sep=""))

  return(list(method=actual.method, ncomp=nzb, b=bvec, res=res))
}
