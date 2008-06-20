## Igamma.inv:
##
## calculate the beta parameter of an Inverse Gamma
## distribution with alpha parameter a at location
## y

Igamma.inv <- function(a, y, lower=FALSE, log=FALSE)
  {
    ## call the C routine
    r <- .C("Igamma_inv_R",
            a = as.double(a),
            y = as.double(y),
            lower = as.integer(lower),
            log = as.integer(log),
            result = double(1),
            PACKAGE = "monomvn")

    return(r$result)
  }


## mvnpdf.C:
##
## function used to test C code for evaluating
## the densift of an MVN distributoin with mean
## mu and covariance S

mvnpdf.C <- function(x, mu, S, log=FALSE)
  {
    r <- .C("mvnpdf_log_R",
       x = as.double(x),
       mu = as.double(mu),
       S = as.double(S),
       n = as.integer(length(x)),
       result =  double(1)
       ,PACKAGE = "monomvn")

    if(log) return(r$result)
    else return(exp(r$result))
  }


## get.regress:
##
## this is the inverse of the monomvn function.
## fiven a monomvn object, this function
## extracts a regession (mu, beta, s2) for each
## component of the mean vector, and each column of the
## covariance matrix

get.regress <- function(x)
  {
    ## put thins into the monomvn order
    mu <- x$mu[x$o]
    S <- x$S[x$o, x$o]
    ncomp <- x$ncomp[x$o]
    na <- x$na[x$o]

    ## initialize a list with first component
    reg <- list()
    reg[[1]] <- list(o=x$o[1], na=na[1], nnew=ncomp[1], ncomp=ncomp[1], mu=mu[1], s2=S[1,1])

    rep <- 0
    
    ## do the rest of the m-1 components
    for(i in 2:length(mu)) {

      ## extract the relevant sub-vectors for the
      ## i-th component
      m1 <- mu[1:(i-1)]
      m2 <- mu[i]
      s11 <- S[1:(i-1), 1:(i-1)]
      s21 <- drop(S[i,1:(i-1)])
      s22 <- drop(S[i, i])

      ## calculate the paramters
      beta <- drop(s21 %*% solve(s11))
      s2 <- drop(s22 - s21 %*% beta) ## do this before zeroing
      
      beta.save <- beta
      if(!is.na(ncomp[i])) {

        if(!is.na(ncomp[i-1])) {
          
          if(na[i] == na[i-1]) {
            rep <- rep + 1
            ext <- sum(abs(beta[(i-rep):(i-1)]) > sqrt(.Machine$double.eps))
            ncomp[i] <- ncomp[i] + ext
          } else rep <- 0
        }
        
        ## to do as the C function does
        q <- as.numeric(quantile(abs(beta), 1.0-ncomp[i]/length(beta)))
        beta[abs(beta) <= q] <- 0

        ## simpler version
        ##beta[abs(beta) < sqrt(.Machine$double.eps)] <- 0
      }
      
      if(s2 <= 0) stop("bad s2")
      b0 <- drop(m2 - beta %*% m1)

      ## add them to the list
      reg[[i]] <- list(o=x$o[i], na=na[i], nnew=sum(beta != 0),
                       ncomp=ncomp[i], mu=b0, beta=beta, s2=s2)
    }

    ## return a list of the regressions
    return(reg)
  }


## get.regress.C:
##
## an R interface to a C version which does the same
## thing as get.regress()

get.regress.C <- function(x)
{
  ## put things into the monomvn order
  mu <- x$mu[x$o]
  S <- x$S[x$o, x$o]
  ncomp <- x$ncomp[x$o]
  na <- x$na[x$o]

  ## initialize a regression list
  reg <- list()

  ## calculate the regression for each of m components
  for(i in 1:length(mu)) {

    if(is.na(ncomp[i])) ncomp[i] <- i-1

    r <- .C("get_regress_R",
            M = as.integer(length(mu)),
            m = as.integer(i),
            mu = as.double(mu),
            S = as.double(S),
            ncomp = as.integer(ncomp[i]),
            b0 = double(1),
            beta = double(i-1),
            s2 = double(1),
            PACKAGE = "monomvn")

    ## special processinf for the first regression
    if(length(r$beta) == 0) r$beta <- NULL

    ## add the i=th component to the list
    reg[[i]] <- list(o=x$o[i], na=na[i], nnew=sum(r$beta != 0),
                     ncomp=ncomp[i], mu=r$b0, beta=r$beta, s2=r$s2)
  }

  ## return a list of the regressions
  return(reg)
}
