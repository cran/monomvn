\name{metrics}
\alias{kl.norm}
\alias{Ellik.norm}
\alias{rmse.muS}

\title{ RMSE, Expected Log Likelihood and KL Divergence Between
        Two Multivariate Normal Distributions }
\description{
  These functions calculate the root-mean-squared-error,
  the expected log likelihood, and Kullback-Leibler (KL) divergence
  (a.k.a. distance), between two multivariate normal (MVN)
  distributions described by their mean vector and covariance matrix
}
\usage{
rmse.muS(mu1, S1, mu2, S2)
Ellik.norm(mu1, S1, mu2, S2, quiet=FALSE)
kl.norm(mu1, S1, mu2, S2, quiet=FALSE, symm=FALSE)
}

\arguments{
  \item{mu1}{ mean vector of first (estimated) MVN }
  \item{S1}{ covariance matrix of first (estimated) MVN }
  \item{mu2}{ mean vector of second (true, baseline, or comparator) MVN }
  \item{S2}{ covariance matrix of second (true, baseline, or comparator) MVN }
  \item{quiet}{ when \code{FALSE} (default) }
  \item{symm}{ when \code{TRUE} a symmetrized version of the
  KL divergence is used; see the note below }
}
\details{
  The root-mean-squared-error is calculated between the entries of
  the mean vectors, and the upper-triangular part of the covariance
  matrices (including the diagonal).
  
  The KL divergence is given by the formula:
  \deqn{D_{\mbox{\tiny KL}}(N_1 \| N_2) = \frac{1}{2}
  \left( \log \left( \frac{|\Sigma_1|}{|\Sigma_2|} \right)
    + \mbox{tr} \left( \Sigma_1^{-1} \Sigma_2 \right) +
    \left( \mu_1 - \mu_2\right)^\top \Sigma_1^{-1}
    ( \mu_1 - \mu_2 ) - N \right)
    }{0.5 (log(|S1|/|S2|) + tr(inv(S1)S2) + t(mu1-m2)inv(S2)(mu1-mu2) - N)}
  
    where \eqn{N}{N} is \code{length(mu1)}, and must agree with
    the dimensions of the other parameters.  Note that the parameterization
    used involves swapped arguments compared to some other references,
    e.g., as provided by Wikipedia.  See note below.

    The expected log likelihood can be formulated in terms of the
    KL divergence.  That is, the expected log likelihood of data
    simulated from the normal distribution with parameters \code{mu2}
    and \code{S2} under the estimated normal with parameters
    \code{mu1} and \code{S1} is given by

  \deqn{ -\frac{1}{2} \ln \{(2\pi e)^N |\Sigma_2|\} -
    D_{\mbox{\tiny KL}}(N_1 \| N_2).
  }{ -0.5 ln((2 pi e)^N |S2|) - kl.norm(mu1, S1, mu2, S2). }
}
\value{
  In the case of the expected log likelihood the result is
  a real number.  The RMSE is a positive real number.
  The KL divergence method returns a positive
  real number depicting the \emph{distance} between the
  two normal distributions
}
\author{ Robert B. Gramacy \email{rbg@vt.edu} }
\note{
  The KL-divergence is not symmetric.  Therefore

  \code{kl.norm(mu1,S1,mu2,S2) != kl.norm(mu2,S2,mu1,S1).}

  But a symmetric metric can be constructed from

  \code{0.5 * (kl.norm(mu1,S1,mu2,S2) + kl.norm(mu2,S2,mu1,S1))}

  or by using \code{symm = TRUE}.  The arguments are reversed
  compared to some other references, like Wikipedia.  To match
  those versions use \code{kl.norm(mu2, S2, mu1, s1)}
}
\references{
\url{http://bobby.gramacy.com/r_packages/monomvn}
}
\examples{
mu1 <- rnorm(5)
s1 <- matrix(rnorm(100), ncol=5)
S1 <- t(s1) \%*\% s1

mu2 <- rnorm(5)
s2 <- matrix(rnorm(100), ncol=5)
S2 <- t(s2) \%*\% s2

## RMSE
rmse.muS(mu1, S1, mu2, S2)

## expected log likelihood
Ellik.norm(mu1, S1, mu2, S2)

## KL is not symmetric
kl.norm(mu1, S1, mu2, S2)
kl.norm(mu2, S2, mu1, S1)

## symmetric version
kl.norm(mu2, S2, mu1, S1, symm=TRUE)
}

\keyword{ multivariate }
