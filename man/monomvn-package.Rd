\name{monomvn-package}
\alias{monomvn-package}
\docType{package}
\title{Estimation for Multivariate Normal and Student-t Data with Monotone Missingness}
\description{Estimation of multivariate normal and student-t data of
arbitrary dimension where the pattern of missing data is monotone.
Through the use of parsimonious/shrinkage regressions
(plsr, pcr, lasso, ridge,  etc.), where standard regressions fail, 
the package can handle a nearly arbitrary amount of missing data.
The current version supports maximum likelihood inference and
a full Bayesian approach employing scale-mixtures for the
lasso (double-exponential) prior and Student-t errors.  Monotone 
data augmentation extends this Bayesian approach to arbitrary
missingness patterns.  A fully functional standalone interface to the 
Bayesian lasso (from Park & Casella) and ridge regression with model 
selection via Reversible Jump, and student-t errors (from Geweke) is 
also provided}
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
Robert B. Gramacy, Joo Hee Lee and Ricardo Silva (2008).
\emph{On estimating covariances between many assets with histories 
of highly variable length}. \cr Preprint available on arXiv:0710.5837:
\url{http://arxiv.org/abs/0710.5837}

\url{http://www.statslab.cam.ac.uk/~bobby/monomvn.html}
}
\keyword{ package }
\seealso{ \code{\link{monomvn}}, \pkg{norm}, \pkg{mvnmle} }
