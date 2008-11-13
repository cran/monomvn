\name{monomvn-package}
\alias{monomvn-package}
\docType{package}
\title{Estimation for Multivariate Normal Data with Monotone Missingness}
\description{
  Estimation of multivariate normal data of arbitrary dimension
  where the pattern of missing data is monotone. Through the use of
  parsimonious/shrinkage regressions (plsr, pcr, lasso, ridge, etc.),
  where standard regressions fail, the package can handle an 
  (almost) arbitrary amount of missing data.  The current version 
  supports maximum likelihood inference and implementation
  of a Bayesian version employing a Bayesian lasso.  A fully 
  functional stand-alone interface to the Bayesian lasso
  (from Park & Casella) and ridge regression with model selection 
  via Reversible Jump is also provided
}
\details{
For a fuller overview including a complete list of functions, demos and
vignettes, please use \code{help(package="monomvn")}.
}
\author{
Robert B. Gramacy \email{bobby@statslab.cam.ac.uk}

Maintainer: 
Robert B. Gramacy \email{bobby@statslab.cam.ac.uk}
}
\references{
Robert B. Gramacy and Joo Hee Lee (2007).
\emph{On estimating covariances between many assets with histories 
of highly variable length}. Preprint available on arXiv:0710.5837:\cr
\url{http://arxiv.org/abs/0710.5837}

\url{http://www.statslab.cam.ac.uk/~bobby/monomvn.html}
}
\keyword{ package }
\seealso{ \code{\link{monomvn}}, \pkg{norm}, \pkg{mvnmle} }
