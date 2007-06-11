\name{posdef.approx}
\alias{posdef.approx}

\title{ Find the Nearest Positive Definite Matrix }
\description{
  Check if the matrix is positive definite via attempted
  Cholesky decomposition, and if not use the
  \code{\link[accuracy]{sechol}} function to get the nearest
  approximate Cholesky decomposition from which to recover an
  approximate covariance matrix which is positive definite.
}
\usage{
posdef.approx(S, name = "S", quiet=FALSE) }
}
\arguments{
  \item{S}{ square \code{matrix} which is supposed to be positive
    definite }
  \item{name}{ name of the matrix -- used for printing errors and
    warnings}
  \item{quiet}{ whether or not to print a \code{warning} when a
    non-positive definite matrix is found }
}
\value{
  If successful, returns a \code{matrix} which is positive definite:
  either the original matrix, or a nearby positive definite approximation.
}
\references{
    Micah Altman, Jeff Gill and Michael McDonald (2003).
  \emph{Numerical Issues in Statistical Computing for the Social
    Scientist.} John Wiley and Sons, New York.
  http://www.hmdc.harvard.edu/micah\_altman/numal/
  
}
\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk} }
\note{
  Requires the \pkg{accuracy} package.  If \code{S} is far from positive
  definite, then the method may fail.
}
\seealso{ \code{link[accuracy]{sechol}} }
\examples{
## most likely generates a matrix which is close to positive definite
## but not quite
s <- matrix(rnorm(100), ncol=5)
S <- (t(s) \%*\% s) + matrix(rnorm(25, sd=10), ncol=5)
S

## a correction, quiet=TRUE so there is no warning printed
posdef.approx(S, quiet=TRUE)
}
\keyword{ multivariate }
