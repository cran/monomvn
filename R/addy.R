## addy:
##
## Use ML to estimate the mean and variance matrix of a
## random vector y=y1,y2 when there is a monotone pattern of
## missing data.  Assume that we have the mean of y1 in m1
## and the variance of y1 in s11, with the variance-covariance
## of the mean vector in c11.  We use the complete cases to
## compute the regression of y2 on y1, then update the mean
## and variance matrix accordingly.  Apply this recursively
## and you can estimate the entire mean vector (with its
## variance matrix) and variance matrix (without its variance
## matrix, unfortunately).
##
## adapted from Daniel F. Heitjan, 03.02.13


`addy` <-
function(y1, y2, m1, s11, method="plsr", p=1.0, ncomp.max=Inf, validation="CV", 
         verb=0, quiet=TRUE)
  {
    ## decide what kind of regression to do and return coeffs & mean-sq resids
    reg <- regress(y1, y2, method, p, ncomp.max, validation, verb, quiet)
    
    ## separate out the intercept term from the regression coeffs
    b0 <- reg$b[1,]
    b1 <- reg$b[-1,]
    s22.1 <- reg$S

    ## Update the parameters

    ## mean
    if(length(m1) == 1) {
      m2 <- b0 + b1 * m1
      s21 <- b1 * s11
    } else {
      m2 <- b0+ t(b1) %*% m1
      s21 <- t(b1) %*% s11
    }

    ## don't actually need to invert s11 here -- check this!
    ## s22 <- s22.1 + s21 %*% solve(s11,t(s21))
    s22 <- s22.1 + t(b1) %*% s11 %*% b1

    ## return
    return(list(method=rep(reg$method, ncol(reg$b)), ncomp=reg$ncomp,
                mu=m2, s21=s21, s22=s22))
  }

