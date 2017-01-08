\name{randmvn}
\alias{randmvn}

\title{ Randomly Generate a Multivariate Normal Distribution }
\description{
  Randomly generate a mean vector and covariance matrix describing
  a multivariate normal (MVN) distribution, and then sample from it
}
\usage{
randmvn(N, d, method = c("normwish", "parsimonious"),
        mup=list(mu = 0, s2 = 1), s2p=list(a = 0.5, b = 1),
        pnz=0.1, nu=Inf)
}

\arguments{
  \item{N}{ number of samples to draw }
  \item{d}{ dimension of the  MVN, i.e., the length of the mean vector
    and the number of rows/cols of the covariance matrix }
  \item{method}{ the default generation method is \code{"norwish"}
    uses the direct method described in the details section below,
    whereas the \code{"parsimonious"} method builds up the random mean
    vector and covariance via regression coefficients, intercepts,
    and variances. See below for more details.  Here, a random number of
    regression coefficients for each regression are set to zero }
  \item{mup}{ a \code{list} with entries \code{$mu} and \code{$s2}:
    \code{$mu} is the prior mean for the independent components
    of the normally distributed mean vector; \code{$s2} is the prior
    variance}
  \item{s2p}{ a \code{list} with entries \code{$a} and \code{$b} 
    only valid for \code{method = "parsimonious"}:
    \code{$a > 0} is the baseline inverse gamma prior scale parameter
    for the regression variances (the actual parameter used for
    each column \code{i in 1:d} of the covariance matrix is
    \code{a + i - 1}); \code{$b >= 0} is the rate parameter}
  \item{pnz}{ a scalar \code{0 <= pnz <= 1}, only valid for
    \code{method = "parsimonious"}: determines the binomial
    proportion of non-zero regression coefficients in the sequential
    build-up of \code{mu} and \code{S}, thereby indirectly determining
    the number of non-zero entries in \code{S}}
  \item{nu}{ a scalar \code{>= 1} indicating the degrees of freedom
    of a Student-t distribution to be used instead of an MVN
    when not infinite }
}
\details{
  In the direct method (\code{"normwish"}) the components of the
  mean vector \code{mu} are iid from a standard normal distribution,
  and the covariance matrix \code{S} is
  drawn from an inverse--Wishart distribution with degrees of freedom
  \code{d + 2} and mean (centering matrix) \code{diag(d)}

  In the \code{"parsimonious"} method \code{mu} and \code{S} are
  built up sequentially by randomly sampling intercepts, regression
  coefficients (of length \code{i-1} for \code{i in 1:d}) and variances
  by applying the \code{monomvn} equations.  A unique prior results
  when a random number of the regression coefficients are set to zero.
  When none are set to zero the direct method results
}
\value{
  The return value is a \code{list} with the following components:
  \item{mu }{randomly generated mean vector of length \code{d} }
  \item{S }{randomly generated covariance \code{matrix} with \code{d}
    rows and \code{d} columns }
  \item{x }{if \code{N > 0} then \code{x} is an \code{N*d}
    \code{matrix} of \code{N} samples from the MVN with mean vector
    \code{mu} and covariance \code{matrix} \code{S}; otherwise when
    \code{N = 0} this component is not included}
}
\author{ Robert B. Gramacy \email{rbg@vt.edu} }
\note{ requires the \code{\link[mvtnorm]{rmvnorm}} function of the
    \pkg{mvtnorm} package }
  \seealso{ \code{\link{rwish}}, \code{\link[mvtnorm]{rmvnorm}},
    \code{\link{rmono}} }
\examples{
randmvn(5, 3)
}
\keyword{ distribution }
\keyword{ datagen }
