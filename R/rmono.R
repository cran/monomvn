## rmono:
##
## takes a complete Nxd data matrix of N observations of d-dim
## random vectors, creates a random monotone missingness pattern
## by replacing entries of x with NA.  The returned x always has
## one complete column, and no column has fewer than m (>=4) non-
## missing entries.  Otherwise, the proportion of missing entries
## in each column can be uniform, or it can have a beta
## distribution with parameters alpha=ab[1] and beta=ab[2]

'rmono' <-
function(x, m=4, ab=NULL)
  {
    N <- nrow(x)
    d <- ncol(x)

    ## check that x is a matrix
    if(is.null(d) || is.null(d)) stop("x not a matrix")

    ## check m
    if(length(m) != 1 || m<4 || m>(N-1) )
      stop("m should be an integer with 4 <= m <= N-1")
    
    ## vector of possible missing entries, N+1 means
    ## none-missing
    miss <- (m+1):(N+1)

    ## uniform monotone missingness
    if(is.null(ab)) chop <- c(sample(miss, d-1, replace=TRUE))
    else {
      ## beta distributed monotone missingness

      ## check ab
      if(length(ab) != 2 || !prod(ab > 0))
        stop("ab should be a positive 2-vector, or NULL")
      
      chop <- c(miss[ceiling(length(miss) * rbeta(d-1, ab[1], ab[2]))])
    }

    ## pick which column is fully observed
    full <- sample(1:ncol(x), 1)

    ## make it so the full column is in the right place
    chop <- c(chop, 0)
    chop[length(chop)] <- chop[full]
    chop[full] <- N+1

    ## chop off the monotone missing data pattern
    for(i in 1:d) {
      if(chop[i] == N+1) next;
      x[chop[i]:N,i] <- NA
    }

    return(x)
  }
