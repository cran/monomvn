\name{rmono}
\alias{rmono}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Randomly Impose a Monotone Missingness Pattern }
\description{
  Randomly impose a monotone missingness pattern by replacing the ends
  of each column of the input matrix by a random number of \code{NA}s
}
\usage{
rmono(x, m = 7, ab = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{ data \code{matrix} }
  \item{m}{ minimum number of non-\code{NA} entries in each column }
  \item{ab}{ a two-vector of \eqn{\alpha}{alpha} (\code{ab[1]}) and
    \eqn{\beta}{beta} (\code{ab[2]}) parameters to a
    Beta\eqn{(\alpha, \beta)}{(alpha, beta)} distribution
    describing the proportion of \code{NA} entries in each column.
    The default setting \code{ab = NULL} yields a uniform distribution }
}
\details{
  The returned \code{x} always has one (randomly selected)
  complete column, and no column has fewer than \code{m}
  non-missing entries.  Otherwise, the proportion of missing entries
  in each column can be uniform, or it can have a beta
  distribution with parameters \eqn{\alpha}{alpha} (\code{ab[1]}) and
  \eqn{\beta}{beta} (\code{ab[2]})
}
\value{
  returns a \code{matrix} with the same dimensions as the input \code{x}
}
\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk}}
\references{
\url{http://www.statslab.cam.ac.uk/~bobby/monomvn.html}
}
\seealso{ \code{randmvn} }

\examples{
out <- randmvn(10, 3)
rmono(out$x)
}

\keyword{ datagen }
