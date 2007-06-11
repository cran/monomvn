## randmvn:
##
## returns N random (x) samples from a randomly generated normal
## distribution via (mu) and (S), also returned.  mu is a
## standard normal vector of d means, and S is an inverse-Wishart
## renerated (dxd) covariance matrix with d+2 degrees of freedom
## and an identity centering matrix (mean)

'randmvn' <-
function(N, d)
  {
    ## check N
    if(length(N) != 1 || N < 0) stop("N should be a nonnegative integer")

    ## check d
    if(length(d) != 1 || N <= 0) stop("d should be a positive integer")
    
    ## load mvtnorm
    if(require(mvtnorm, quietly=TRUE) == FALSE)
      stop("this function requires library(mvtnorm)")

    ## generate a coavariance matrix and mean vector
    S <- solve(rwish(d+2, diag(d)))
    mu <- rnorm(d)

    ## don't draw if N=0
    if(N == 0) return(list(mu=mu, S=S))

    ## draw N samples from the MVN
    x <- rmvnorm(N, mu, S)
    return(list(x=x, mu=mu, S=S))
  }
