\name{kl.norm}
\alias{kl.norm}

\title{ KL Divergence Between Two Multivariate Normal Distributions }
\description{
  Returns the Kullback-Leibler (KL) divergence (a.k.a. distance) between two
  multivariate normal (MVN) distributions described by their mean vector and
  covariance matrix
}
\usage{
kl.norm(mu1, S1, mu2, S2, quiet=FALSE)
}

\arguments{
  \item{mu1}{ mean vector of first MVN }
  \item{S1}{ covariance matrix of first MVN }
  \item{mu2}{ mean vector of second MVN }
  \item{S2}{ covariance matrix of second MVN }
  \item{quiet}{ when \code{FALSE} (default), gives a warning if
    \code{\link{posdef.approx}} finds a non-positive definite
    covariance matrix}
}
\details{
  The KL divergence is given by the formula:
  \deqn{
  D_{\mbox{\tiny KL}}(N_1 \| N_2) = \frac{1}{2}
  \left( \log \left( \frac{|\Sigma_2|}{|\Sigma_1|} \right)
    + \mbox{tr} \left( \Sigma_2^{-1} \Sigma_1 \right) +
    \left( \mu_2 - \mu_1\right)^\top \Sigma_2^{-1}
    ( \mu_2 - \mu_1 ) - N \right)
    }{0.5 (log(det(S2)/det(S1)) + tr(inv(S2)S1) +
      t(mu2-m1)inv(S2)(mu2-mu1) - N)}
  
  where \eqn{N}{N} is \code{length(mu1)}.

  As a preprocessing step \code{S1} and \code{S2} are checked
  to be positive definite via the \code{\link{posdef.approx}}.  If
  not, and if the \pkg{accuracy} package is installed, then
  they can be coerced to the nearest positive definite matrix.
}
\value{
  Returns a positive real number giving the KL divergence between the
  two normal distributions.
}
\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk} }
\note{
  The KL-divergence is not symmetric.  Therefore

  \code{kl.norm(mu1,S1,mu2,S2) != kl.norm(mu2,S2,mu1,S1).}

  But a symmetric metric can be constructed from

  \code{0.5 * (kl.norm(mu1,S1,mu2,S2) + kl.norm(mu2,S2,mu1,S1))}
}
\references{
\url{http://www.statslab.cam.ac.uk/~bobby/monomvn.html}
}
\seealso{ \code{\link{posdef.approx}} }
\examples{
mu1 <- rnorm(5)
s1 <- matrix(rnorm(100), ncol=5)
S1 <- t(s1) \%*\% s1

mu2 <- rnorm(5)
s2 <- matrix(rnorm(100), ncol=5)
S2 <- t(s2) \%*\% s2

## not symmetric
kl.norm(mu1,S1,mu2,S2)
kl.norm(mu2,S2,mu1,S1)

## symmetric
0.5 *(kl.norm(mu1,S1,mu2,S2) + kl.norm(mu2,S2,mu1,S1))
}

\keyword{ multivariate }
