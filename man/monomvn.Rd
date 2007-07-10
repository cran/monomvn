\name{monomvn}
\alias{monomvn}
\alias{print.monomvn}

\title{ Estimation of Multivariate Normal Data with Monotone Missingness }

\description{
Maximum likelihood estimation of the mean and covariance matrix of
multivariate normal (MVN) distributed data with a monotone missingness pattern.
Through the use of partial least squares and principal component
regressions, where standard regressions fail,
this function can handle an (almost) arbitrary amount
of missing data.
}
\usage{
monomvn(y, pre = TRUE, method = c("plsr", "pcr"), p = 0.9, 
	ncomp.max = Inf, validation = c("CV", "LOO"), obs = FALSE, 
        verb = 0, quiet = TRUE)
}

\arguments{
  \item{y}{ data \code{matrix}  were each row is interpreted as a
    random sample from a MVN distribution with missing
    values indicated by \code{NA}}
  \item{pre}{ logical indicating whether pre-processing of the
    \code{y} is to be performed.  This sorts the columns so that the
    number of \code{NA}s is non-decreasing with the column index }
  \item{method}{ describes the type of \emph{parsimonious} regression to
    be performed when standard least-squares regression fails.
    \code{"plsr"} (the default) is partial least squares and \code{"pcr"}
    is standard principal component regression.  See the
    \pkg{pls} package, particularly the functions
    \link[pls]{plsr} and \link[pls]{pcr}}
  \item{p}{ when performing regressions, \code{p} is the proportion of the
    number of columns to rows in the design matrix before an
    alternative regression \code{method} (those above) is performed as if
    least-squares regression \dQuote{failed}.  Least-squares regression is
    known to fail when the number of columns equals the number of rows,
    hence a default of \code{p = 0.9} close to \code{1}.
    Alternatively, setting \code{p = 0}
    forces \code{method} to be used for \emph{every} regression.
    Intermediate settings of \code{p} allow the user to control when
    least-squares regressions stop and the \code{method} ones start}
  \item{ncomp.max}{ maximal number of (principal) components to include
    in a \code{method}.  Large settings can cause the execution to be
    slow as it drastically increases the cross-validation (CV) time}
  \item{validation}{ method for validation when applying 
  \code{\link[pls]{plsr}} or \code{\link[pls]{pcr}}.  The default setting
  of \code{"CV"} (randomized 10-fold cross-validation) is the faster method, 
  but does not yield a deterministic result and does not apply for regressions
  on less than ten responses.  \code{"LOO"} (leave-one-out cross-validation)
  is deterministic, always applicable, and applied automatically whenever 
  \code{"CV"} cannot be used}
  \item{obs}{ logical indicating whether or not to (additionally)
    compute a mean vector and covariance matrix based only on the observed
    data, without regressions.  I.e., means are calculated via averages of each
    non-\code{NA} entry in each column of \code{y}, and entries
    \code{(a,b)} of the
    covariance matrix are calculated by applying \code{\link{cov}(ya,yb)}
    to the jointly non-\code{NA} entries of columns \code{a} and \code{b}
    of \code{y}}
  \item{verb}{ whether or not to print progress indicators.  The default
    (\code{verb = 0}) keeps quiet, while any positive number causes brief
    statement about dimensions of each regression to print to
    the screen as it happens.  Future versions will allow \code{verb}
    more granular control of the level of verbosity}
  \item{quiet}{ causes \code{\link{warning}}s about regressions to be silenced
    when \code{TRUE}}
}
\details{
  If \code{pre = TRUE} then \code{monomvn} first re-arranges the columns
  of \code{y} into nondecreasing order with respect to the number of
  missing (\code{NA})  entries. Then (at least) the first column should
  be completely observed.  The mean components and covariances between
  the first set of complete columns are obtained through the standard
  \code{\link{mean}} and \code{\link{cov}} routines.

  Next each successive group of column with the same missingness pattern
  is processed in sequence.  Suppose a total of \code{j} columns have
  been processed this way already.  Let \code{y2} represent the non-missing
  contingent of the next group of \code{k} columns of \code{y}
  with identical missingness pattern, and let \code{y1} be the
  previously processed \code{j-1} columns of \code{y}
  containing only the rows
  corresponding to each non-\code{NA} entry in \code{y2}.  I.e.,
  \code{nrow(y1) = nrow(y2)}.  Note that \code{y1} contains no
  \code{NA} entries since the missing data pattern is monotone.
  The \code{k} next entries (indices \code{j:(j+k)}) of the mean vector,
  and the \code{j:(j+k)} rows and columns of the covariance matrix are
  obtained by multivariate regression of \code{y2} on \code{y1}.
  The regression method used depends on the number of rows and columns
  in \code{y1} and on the \code{p} parameter.  Whenever \code{ncol(y1)
    < p*nrow(y1)} least-squares regression is used, otherwise
  \code{method = c("pcr", "plsr")}.  If ever a least-squares regression
  fails due to co-linearity the one of the other \code{method}s is tried.
  
  Both \code{method}s require a scheme for estimating the amount of
  variability explained by increasing numbers of (principal) components.
  Towards this end, the \pkg{pls} package supports either cross
  validation (CV) or leave-one-out (LOO) estimates of root mean squared
  error (ERROR).  See \pkg{pls} for more details.  \code{monomvn} uses
  CV in all cases except when \code{nrow(y1) <= 10}, in which case CV fails and
  LOO is used.  Whenever \code{nrow(y1) <= 3} \code{\link[pls]{pcr}}
  fails,  so \code{\link[pls]{plsr}} is used instead.
  If \code{quiet = FALSE} then a \code{\link{warning}}
  is given whenever the first choice for a regression fails.  RMSEs
  are calculated for a number of components in \code{1:ncomp.max} where
  \code{is.null(ncomp.max)} it is replaced with

  \code{ncomp.max <- min(ncomp.max, ncol(y2), nrow(y1)-1)}

  which is the max allowed by the \pkg{pls} package.
  A simple heuristic is used to select a small number of components
  (\code{ncomp}) which explains a large amount of the variability (RMSE).

  Based on the MLE \code{ncol(y1)+1} regression coefficients (including
  intercept)
  obtained for each of the
  columns of \code{y2}, and on the corresponding \code{matrix} of
  residual sum of squares, and on the previous \code{j-1} means
  and rows/cols of the covariance matrix, the \code{j:(j+k)} entries and
  rows/cols can be filled in as described by Little and Rubin, section 7.4.3.

  Once every column has been processed the entries of the mean vector, and
  rows/cols of the covariance matrix are re-arranged into their original
  order.
}
\value{
  \code{monomvn} returns an object of class \code{"monomvn"}, which is a
  list containing a subset of the components below.

  \item{call }{a copy of the function call as used}
  \item{mu }{estimated mean vector with columns corresponding to the
    columns of \code{y} }
  \item{S }{estimated covariance matrix with rows and columns
    corresponding to the columns of \code{y} }
  \item{na }{ when \code{pre = TRUE} this is a vector containing number of
    \code{NA} entries in each column of \code{y}}
  \item{o }{ when \code{pre = TRUE} this is a vector containing the
    index of each column in the sorting of the columns of \code{y}
    obtained by \code{o <- \link{order}(na)}}
  \item{method }{method of regression used on each column, or
    \code{"complete"} indicating that no regression was necessary}
  \item{ncomp }{number of components in a \code{\link[pls]{plsr}} or
    \code{\link[pls]{pcr}} regression, or \code{NA} such a method was
    not used}
  \item{mu.obs }{when \code{obs = TRUE} this is the \dQuote{observed}
    mean vector}
  \item{S.obs }{when \code{obs = TRUE} this is the \dQuote{observed}
    covariance matrix, as described above.  Note that \code{S.obs} is
    usually not positive definite }
}
\references{
  Roderick J.A. Little and Donald B. Rubin (2002).
  \emph{Statistical Analysis with Missing Data}, Second Edition.
  Wilely.

  Bjorn-Helge and Mevik Ron Wehrens (2007).
  \emph{The \pkg{pls} Package: Principal Component and Partial
    Least Squares Regression in R.} 
  Journal of Statistical Software \bold{18}(2)
  
  Some of the code for \code{monomvn}, and its subroutines, was inspired
  by code found on the world wide web, written by Daniel Heitjan,
  \url{http://www.cceb.upenn.edu/pages/heitjan/courses/bsta782/examples/fcn.q}
}
\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk}

}
\note{
  Note that the covariances uses throughout are the MLE, i.e., not
  unbiased ones which have an \code{N-1} in the denominator

  The CV in \pkg{plsr} is random in nature, so whenever
  \code{\link[pls]{plsr}} or \code{\link[pls]{pcr}} is used the results
  can be dependent on the random seed
}
\seealso{ \link[norm]{em.norm} in the \pkg{norm} package, and
  \link[mvnmle]{mlest} in the \pkg{mvnmle} package}
\examples{
## standard usage, duplicating the results in
## Little and Rubin, section 7.4.3
data(cement.miss)
out <- monomvn(cement.miss)
out
out$mu
out$S

##
## A bigger example, comparing plsr to pcr
##

## generate N=1000 samples from a random MVN
xmuS <- randmvn(1000, 100)

## randomly impose monotone missingness
xmiss <- rmono(xmuS$x)

## plsr
out.plsr <- monomvn(xmiss)
out.plsr
kl.norm(out.plsr$mu, out.plsr$S, xmuS$mu, xmuS$S)

## plcr
out.pcr <- monomvn(xmiss, method="pcr")
out.pcr
kl.norm(out.pcr$mu, out.pcr$S, xmuS$mu, xmuS$S)
}

\keyword{ multivariate }
\keyword{ regression }
