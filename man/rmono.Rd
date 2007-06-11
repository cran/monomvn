\name{rmono}
\alias{rmono}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Randomly Impose a Monotone Missingness Pattern }
\description{
  Randomly impose a monotone missingness pattern by replacing the ends
  of each column of the input matrix by a random number of \code{NA}s
}
\usage{
rmono(x, m = 4, ab = c(1, 2))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{ data \code{matrix} }
  \item{m}{ minimum number of non-\code{NA} entries in each column }
  \item{ab}{ a two-vector of \eqn{\alpha}{alpha} (\code{ab[1]}) and
    \eqn{\beta}{beta}
    (\code{\ab[2]}) parameters to a beta distribution describing the
      proportion of \code{NA} entries in each column.  If \code{ab =
	NULL} then a uniform distribution is used }
}
\details{
The returned \code{x} always has
one complete column, and no column has fewer than \code{m >= 4}
non-missing entries.  Otherwise, the proportion of missing entries
in each column can be uniform, or it can have a beta
distribution with parameters \eqn{\alpha}{alpha} (\code{ab[1]}) and
\eqn{\beta}{beta} (\code{ab[2]})
}
\value{
  returns a matrix with the same dimensions as the input \code{x}
}
\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk}}
\seealso{ \code{randmvn} }

\examples{
out <- randmvn(10, 3)
rmono(out$x)
}

\keyword{ datagen }
