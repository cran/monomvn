## monomvn:
##
## Under a MVN model, calculate the MLE of the mean and variance
## matrix from a data matrix y that is potentially subject to
## monotone missingness.  We first compute the mean and variance
## of column 1 (assumed to be complete), then successively add
## columns one at a time until it's finished, regardless of the
## completeness of the successive columns.  The rows need not be
## sorted to give a monotone appearance, but the pattern must be
## monotone.
##
## adapted from Daniel Heitjan, 03.02.14


`monomvn` <-
function(y, pre=TRUE,
         method=c("plsr", "pcr", "lasso", "lar", "forward.stagewise",
           "stepwise"), p=0.9, ncomp.max=Inf,
         validation=c("CV", "LOO"), obs=FALSE, verb=0, quiet=TRUE)
  {
    ##
    ## Begin: pre-processing
    ##

    ## check method argument
    method <- match.arg(method)
    
    ## check p argument
    if(length(p) != 1 || p > 1 || p < 0) {
      warning("should have scalar 0 <= p <= 1, using default p=1")
      p <- 1
    }

    ## check ncomp.max argument
    if(length(ncomp.max) != 1 || ncomp.max < 1) {
      warning("should have integer 1 <= ncomp.max, using default ncomp=Inf")
      ncomp.max <- Inf
    }

    ## check validation argument
    validation <- match.arg(validation)

    ## check pre argument
    if(length(pre) != 1 || !is.logical(pre))
      stop("pre should be a logical scalar")

    ## check obs argument
    if(length(obs) != 1 || !is.logical(obs))
      stop("obs should be a logical scalar")

    ## check verb argument
    if(length(verb) != 1 || verb < 0)
      stop("verb should be a positive scalar")

    ## save the call
    cl <- match.call()
    
    ## save column namings in a data frame, and then work with a matrix
    nam <- colnames(y)
    y <- as.matrix(y)

    ## imput dimension and checks
    n <- nrow(y)
    m <- ncol(y)
    if (m < 2) stop('Need at least two columns')

    ## get the number of nas in each column
    nas <- apply(y, 2, function(x) {sum(is.na(x))} )
    
    ## check for cols with all NAs
    if(sum(nas == n) > 0) {
      cat("cols with no data:\n")
      print((1:m)[nas == n])
      stop("remove these columns and try again")
    }
    
    ## re-order the columns to follow the monotone pattern
    if(pre) {
      nao <- order(nas)
      y <- y[,nao]
    } else {
      nao <- 1:ncol(y)
    }

    ## get indices where the missingness pattern changes,
    ## in particular where the missingness increases
    naso <- nas[nao]
    miss <- (1:m)[duplicated(naso) == FALSE]
    miss <- c(miss, m+1)

    ## for holding the means and covars
    mu <- rep(0, m)
    S <- matrix(0, ncol=m, nrow=m)
    
    ## for holding the observed means and covars
    if(obs) {
      mu.obs <- rep(0, m)
      S.obs <- matrix(0, ncol=m, nrow=m)
    }

    ## for saving a trace of which regression methods were used
    methods <- rep("complete", length=ncol(y))
    ncomp <- rep(NA, length=ncol(y))
    
    ## First make sure column 1 is complete.
    touse <- 1*(!is.na(y[,1]))
    if (min(touse)==0) stop('first column is not complete')
    
    ## find how many of the initial columns are complete
    na <- apply(y, 2, function(x) {sum(is.na(x))})
    fcol <- miss[1]:(miss[2]-1)
    if(fcol[length(fcol)] != length(fcol))
      stop("missingness pattern not monotone")

    ##
    ## End: pre-processing
    ## Begin: calculations based on complete data rows
    ##
    
    ## Mean and variance for the first length(fcol) complete columns
    mu[fcol] <- matrix(apply(as.matrix(y[,fcol]),2,mean),length(fcol),1)
    S[fcol,fcol] <- matrix((n-1)*cov(as.matrix(y[,fcol]))/n,length(fcol),length(fcol))

    ## save the observed means and covariances
    if(obs) {
      mu.obs[fcol] <- mu[fcol]
      S.obs[fcol,fcol] <- S[fcol,fcol]
    }
    
    ##
    ## End: calculations based on complete data rows
    ## Begin: processing rows with increasing missingness
    ##
    
    ## no missing data
    if(length(fcol) == m) {
      if(!quiet) warning("no missing data")
      
    } else { ## yes, there is missing data, so we need to do regressions
    
      ## Now loop through the remaining columns, adding them one by one.
      ## for (j in (length(fcol)+1):m) {
      for (i in 2:(length(miss)-1)) {

        ## First check each of the next group of columns for monotonicity
        ## (MIGHT BE A BETTER WAY TO DO THIS)
        for(j in miss[i]:(miss[i+1]-1)) {
          jm1 <- j-1
          tousejm1 <- !is.na(y[,jm1])
          tousej <- !is.na(y[,j])
          diff <- as.numeric(tousej) - as.numeric(tousejm1)
          if(max(diff) > 0) stop(paste("variable", nao[j], "violates monotonicity"))
        }

        ## also check for minumum number of variables
        ##if(sum(tousej) <= 3)
          ##stop(paste("variable ", nao[j], " has less than 4 non-NA entries", sep=""))

        ## a are column indices processed alread, b are new indices
        ## y1 are old columns, and y2 is new column
        a <- 1:(miss[i]-1);  b <- miss[i]:(miss[i+1]-1)
        y1<-y[tousej,a]; y2<-y[tousej,b]
        
        if(obs) {
          ## save the observed means
          mu.obs[b] <- apply(as.matrix(y2), 2, mean)
        
          ## more rows/cols to the observed covariance matrix
          lty <- nrow(y1); if(is.null(lty)) lty <- 1
          S.obs[a,b] <- (lty-1)*cov(y1,y2)/lty
          S.obs[b,a] <- t(S.obs[a,b])
          S.obs[b,b] <- (lty-1)*cov(as.matrix(y2))/lty
        }
        
        ## regress to add a component to mu, and a row/col to S
        add <- addy(y1, y2, mu[a], S[a,a], method, p, ncomp.max, validation, 
                    verb, quiet)
        if(verb > 0) cat("\n")
        
        ## save updated mu and S, and other stats from addy
        mu[b] <- add$mu
        S[b,a] <- add$s21
        S[a,b] <- t(S[b,a])
        S[b,b] <- add$s22
        methods[b] <- add$method
        ncomp[b] <- add$ncomp
      }
    }

    ##
    ## End: processing rows with increasing missingness
    ## Begin: post-processing
    ##
    
    ## put the original ordering back
    if(pre) {
      oo <- order(nao)
      mu <- mu[oo]
      S <- S[oo,oo]
      methods <- methods[oo]
      ncomp <- ncomp[oo]

      ## do the same for obs, if calculated
      if(obs) {
        mu.obs <-mu.obs[oo]
        S.obs <- S.obs[oo,oo]
      }
    }
    
    ## deal with names
    if(! is.null(nam)) {
      mu <- matrix(mu, nrow=length(mu))
      rownames(mu) <- nam
      colnames(S) <- rownames(S) <- nam
      
      ## do the same for obs, if calculated
      if(obs) {
        mu.obs <- matrix(mu.obs, nrow=length(mu))
        rownames(mu.obs) <- nam
        colnames(S.obs) <- rownames(S.obs) <- nam
      }
      
    } else {
      ## otherwise, let mu return as a vector rather than data.frame
      mu <- as.vector(mu)
    }

    ## done, make initial class-list for return
    r <- list(call=cl, methods=methods, ncomp=ncomp, mu=mu, S=S)

    ## possibly add column permutation info from pre-processing
    if(pre) {
      r$na <- nas
      r$o <- nao
    }

    ## possibly add observed-data calculations
    if(obs) {
      r$mu.obs <- mu.obs
      r$S.obs <- S.obs
    }

    ## assign class and return
    class(r) <- "monomvn"
    return(r)
  }

