\name{cement}
\alias{cement}
\alias{cement.miss}
\docType{data}
\title{Hald's Cement Data}
\description{
Heat evolved in setting of cement, as a function of its chemical
composition.
}
\usage{
data(cement)
data(cement.miss)
}
\format{
  A data frame with 13 observations on the following 5 variables.
  \describe{
    \item{x1 }{percentage weight in clinkers of 3CaO.Al2O3}
    \item{x2 }{percentage weight in clinkers of 3CaO.SiO2}
    \item{x3 }{percentage weight in clinkers of 4CaO.Al2O3.Fe2O3}
    \item{x4 }{percentage weight in clinkers of 2CaO.SiO2}
    \item{y }{heat evolved (calories/gram)}
  }
}
\details{
  \code{cement.miss} is taken from an example in Little & Rubin's book
  on \emph{Statistical Analysis with Missing Data} (2002), pp. 154, for
  demonstrating estimation of multivariate means and variances when
  the missing data pattern is monotone.  These are indicated by
  \code{NA} in \code{cement.miss}.  See the examples section of
  \code{\link{monomvn}} for a re-working of the example from the textbook.
}
\source{
Woods, H., Steinour, H. H. and Starke, H. R. (1932) 
Effect of composition of Portland cement on heat evolved during hardening.
\emph{Industrial Engineering and Chemistry}, \bold{24}, 1207--1214.
}
\references{
Davison, A. C. (2003)  \emph{Statistical Models}.  Cambridge University Press.
Page 355.

Draper, N.R. and Smith, H. (1998) \emph{Applied Regression
  Analysis}. Wiley. Page 630.

Roderick J.A. Little and Donald B. Rubin (2002).
\emph{Statistical Analysis with Missing Data}, Second Edition.
Wilely.  Page 154.

\url{http://www.statslab.cam.ac.uk/~bobby/monomvn.html}
}

\seealso{
  \code{\link{monomvn}} --  Several other \R packages also include
    this data set.
  }

\examples{
data(cement)
lm(y~x1+x2+x3+x4,data=cement)
}
\keyword{datasets}
