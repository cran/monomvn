## regress.ls:
##
## fit y2 ~ y1  using LM (i.e., least squares)


'regress.ls' <-
  function(y1, y2, verb=0)
{
  ## number of regressions
  numreg <- ncol(y2); if(is.null(numreg)) numreg <- 1
  
  ## add to progress meter
  if(verb > 0) cat(paste("using lsr ", sep=""))
  
  ## standard least-squares regression
  reglst <- lm(y2 ~ y1)
  bvec <- matrix(reglst$coef, ncol=numreg)
  res <- matrix(reglst$resid, ncol=numreg)
  actual.method <- "lsr"
  ncomp <- rep(NA, numreg)
  
  return(list(method=actual.method, ncomp=ncomp, b=bvec, res=res))
}
