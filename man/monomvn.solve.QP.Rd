\name{monomvn.solve.QP}
\alias{monomvn.solve.QP}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Solve a Quadratic Program  }
\description{
  Solve a Quadratic Program specified by a \code{QP} object
  using the covariance matrix and mean vector specified
}
\usage{
monomvn.solve.QP(S, QP, mu = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{S}{ a positive-definite covariance \code{matrix} whose
    dimensions agree with \code{nrow(QP$Amat)} }
  \item{QP}{ a Quadratic Programming object like one that can
    be generated automatically by \code{default.QP} }
  \item{mu}{ an mean vector with
    \code{length(mu) = nrow(QP$Amat)} that is required
    if \code{QP$dmu == TRUE} or \code{QP$mu.constr[1] != 0}}
}
\details{
  The protocol executed by this function is identical to
  the one used on samples of \eqn{\Sigma}{S} and \eqn{\mu}{mu}
  obtained in \code{\link{bmonomvn}} when a Quadratic Program
  is specified through the \code{QP} argument.  For more details
  on the specification of the Quadratic Program implied by a
  \code{QP} object, please see \code{\link{default.QP}} and
  examples therein
}
\value{
  The output is a vector with length agreeing with
  the dimension of \code{S} describing the solution to the
  Quadratic Program given
}
%\references{ ~put references to the literature/web site here ~ }
\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk} }
\seealso{ \code{\link{default.QP}}, \code{\link{bmonomvn}},
  and \code{\link[quadprog]{solve.QP}} in the \pkg{quadprog} package
  }
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ optimize }
