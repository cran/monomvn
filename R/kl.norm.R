## kl.norm
##
## Kulback-Leibler divergence between two multivariate normal
## distributions specified by their mean vectors and
## covariance matrices -- if a covariance matrix is not
## positive definite, then the nearst posdef one is found and
## used in the distance calculation

`kl.norm` <-
function(mu1, S1, mu2, S2, quiet=FALSE)
{
  N <- length(mu1)

  ## check that the mean vectors have the same length
  if(length(mu2) != N) stop("must have length(mu1) == length(mu2)")

  ## check the covar matrices have same dims as the mean
  if(ncol(S1) != N || nrow(S1) != N)
    stop("must have nrow(S1) == ncol(S1) == length(mu1)")
  if(ncol(S2) != N || nrow(S2) != N)
    stop("must have nrow(S2) == ncol(S2) == length(mu2)")

  ## force positive definiteness of the covs
  S1 <- posdef.approx(S1, "S1", quiet)
  S2 <- posdef.approx(S2, "S2", quiet)

  ##
  ## distance calculation in parts
  ##

  ## calculate the determinants in log space
  ld2 <- determinant(S2, logarithm=TRUE)
  if(ld2$sign == -1) stop("S2 is not posdef")
  ld1 <- determinant(S1, logarithm=TRUE)
  if(ld1$sign == -1) stop("S1 is not posdef")
  ldet <- ld2$modulus[1] - ld1$modulus[1]

  ## rest of the calculation
  S2i <- solve(S2)
  tr <- sum(diag(S2i %*% S1))
  m2mm1 <- mu2 - mu1
  qf <- as.numeric(t(m2mm1) %*% S2i %*% m2mm1)

  ## return the correct combination of the parts
  return(0.5*(ldet + tr + qf - N))
}

