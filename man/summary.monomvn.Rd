\name{summary.monomvn}
\alias{summary.monomvn}
\title{ Summarizing monomvn output }
\description{
  Simply calls \code{\link{print.monomvn}} on the \code{"monomvn"}-class
  object
}
\usage{
\method{summary}{monomvn}(object, ...)
}
\arguments{
  \item{object}{ a \code{"monomvn"}-class object }
  \item{...}{ passed to \code{\link{print.monomvn}}}
}
\details{
  Prints the \code{call} followed by a summary of the regression
  method used at each iteration of the algorithm, also indicating how
  many completely observed features (columns) there were in the data.
  For least-squares regressions (i.e., \code{"plsr"} an \code{"pcr"})
  and indication of the method used for selecting the
  number of components (i.e., \code{CV}, \code{LOO}, or \code{none})
}
\value{
  For now, simply returns the text of \code{\link{print.monomvn}}
}
\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk} }
\references{
\url{http://www.statslab.cam.ac.uk/~bobby/monomvn.html}
}
\seealso{ \code{\link{monomvn}} }
}
\keyword{ methods }
