\name{posdef.approx}
\alias{posdef.approx}

\title{ Find the Nearest Positive Definite Matrix }
\description{
  Check if the matrix is positive definite via attempted
  Cholesky decomposition and inversion.  If not, then the
  \code{\link[accuracy]{sechol}} function is used to obtain the nearest
  approximate Cholesky decomposition from which to recover an
  approximate covariance matrix which is positive definite.
  If that doesn't do the trick, then \code{.Machine$double.eps} is
  added to the diagonal until a positive definite matrix can be returned
}
\usage{
posdef.approx(S, name = "S", quiet=FALSE)
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
  Returns a \code{matrix} which is positive definite:
  either the original matrix, or a nearby positive definite approximation.
}
\references{
    Micah Altman, Jeff Gill and Michael McDonald (2003).
  \emph{Numerical Issues in Statistical Computing for the Social
    Scientist.} John Wiley and Sons, New York.\cr
  \url{http://www.hmdc.harvard.edu/micah\_altman/numal/}

\url{http://www.statslab.cam.ac.uk/~bobby/monomvn.html}
}
\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk} }

\seealso{ \code{\link[accuracy]{sechol}} in the \pkg{accuracy} package }
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
