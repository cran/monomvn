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
a full Bayesian approach employing scale-mixtures for Gibbs sampling.
Monotone data augmentation extends this Bayesian approach to arbitrary
missingness patterns.  A fully functional standalone interface to the 
Bayesian lasso (from Park & Casella), the Normal-Gamma (from Griffin
& Brown), Horseshoe (from Carvalho, Polson, & Scott), and ridge regression with model 
selection via Reversible Jump, and student-t errors (from Geweke) is 
also provided}
\details{
For a fuller overview including a complete list of functions, demos and
vignettes, please use \code{help(package="monomvn")}.
}
\author{
Robert B. Gramacy \email{rbg@vt.edu}

Maintainer: 
Robert B. Gramacy \email{rbg@vt.edu}
}
\references{
Robert B. Gramacy, Joo Hee Lee and Ricardo Silva (2008).
\emph{On estimating covariances between many assets with histories 
of highly variable length}. \cr Preprint available on arXiv:0710.5837:
\url{https://arxiv.org/abs/0710.5837}

\url{https://bobby.gramacy.com/r_packages/monomvn/}
}
\keyword{ package }
\seealso{ \code{\link{monomvn}}, the now defunct \code{norm} package, \pkg{mvnmle} }
