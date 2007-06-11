\name{randmvn}
\alias{randmvn}

\title{ Randomly Generate a Multivariate Normal Distribution }
\description{
  Randomly generate a mean vector and covariance matrix describing
  a multivariate normal (MVN) distribution, and then sample from it
}
\usage{
randmvn(N, d)
}

\arguments{
  \item{N}{ number of samples to draw }
  \item{d}{ dimension of the  MVN, i.e., the length of the mean vector
    and the number of rows/cols of the covariance matrix }
}
\details{
  The components of the mean vector \code{mu} are iid from a
  standard normal distribution, and the covariance matrix \code{S} is
  drawn from an inverse--Wishart distribution with degrees of freedom
  \code{d + 2} and mean (centering matrix) \code{diag(d)}
}
\value{
  The return value is a \code{list} with the following components:
  \item{mu }{randomly generated mean vector of length \code{d} }
  \item{S }{randomly generated covariance \code{matrix} with \code{d}
    rows and \code{d} columns }
  \item{x }{if \code{N > 0} then \code{x} is an \code{N} by \code{d}
    \code{matrix} of \code{N} samples from the MVN with mean vector
    \code{mu} and covariance \code{matrix} \code{S}; otherwise when
    \code{N = 0} this component is not included}
}
\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk} }
\note{ requires the \code{\link[mvtnorm]{rmvnorm}} function of the
    \pkg{mvtnorm} package }
\seealso{ \code{\link{rwish}}, \code{\link[mvtnorm]{rmvnorm}}, \code{\link{rmono}} }
\examples{
randmvn(5, 3)
}
\keyword{ distribution }
\keyword{ datagen }
