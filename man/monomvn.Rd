\name{monomvn}
\alias{monomvn}

\title{ Maximum Likelihood Estimation for Multivariate Normal
  Data with Monotone Missingness }

\description{
Maximum likelihood estimation of the mean and covariance matrix of
multivariate normal (MVN) distributed data with a monotone missingness pattern.
Through the use of parsimonious/shrinkage regressions (e.g., plsr, pcr,
ridge, lasso, etc.), where standard regressions fail,
this function can handle an (almost) arbitrary amount of missing data
}
\usage{
monomvn(y, pre = TRUE, method = c("plsr", "pcr", "lasso", "lar",
        "forward.stagewise", "stepwise", "ridge", "factor"), p = 0.9,
        ncomp.max = Inf, batch = TRUE, validation = c("CV", "LOO", "Cp"),
        obs = FALSE, verb = 0, quiet = TRUE)
}

\arguments{
  \item{y}{ data \code{matrix}  were each row is interpreted as a
    random sample from a MVN distribution with missing
    values indicated by \code{NA}}
  \item{pre}{ logical indicating whether pre-processing of the
    \code{y} is to be performed.  This sorts the columns so that the
    number of \code{NA}s is non-decreasing with the column index }
  \item{method}{ describes the type of \emph{parsimonious}
    (or \emph{shrinkage}) regression to
    be performed when standard least squares regression fails.
    From the \pkg{pls} package we have \code{"plsr"}
    (\link[pls]{plsr}, the default) for  partial least squares and
    \code{"pcr"} (\link[pls]{pcr}) for standard principal
    component regression.  From the \pkg{lars} package (see the
    \code{"type"} argument to \link[lars]{lars})
    we have \code{"lasso"} for L1-constrained regression, \code{"lar"}
    for least angle regression, \code{"forward.stagewise"} and
    \code{"stepwise"} for fast implementations of classical forward
    selection of covariates.  From the \pkg{MASS} package we have
    \code{"ridge"} as implemented by the \code{\link[MASS]{lm.ridge}}
    function.  The \code{"factor"} method treats the first \code{p}
    columns of \code{y} as known factors }
  \item{p}{ when performing regressions, \code{p} is the proportion of the
    number of columns to rows in the design matrix before an
    alternative regression \code{method} (those above) is performed as if
    least-squares regression has \dQuote{failed}.  Least-squares regression is
    known to fail when the number of columns equals the number of rows,
    hence a default of \code{p = 0.9 <= 1}. Alternatively, setting
    \code{p = 0} forces \code{method} to be used for \emph{every} regression.
    Intermediate settings of \code{p} allow the user to control when
    least-squares regressions stop and the \code{method} ones start.
    When \code{method = "factor"} the \code{p} argument represents an
    integer (positive) number of initial columns of \code{y} to treat
    as known factors }
  \item{ncomp.max}{ maximal number of (principal) components to include
    in a \code{method}---only meaningful for the \code{"plsr"} or
    \code{"pcr"} methods.  Large settings can cause the execution to be
    slow as it drastically increases the cross-validation (CV) time}
  \item{batch}{ indicates whether the columns with equal missingness
    should be processed together using a multi-response regression.
    This is more efficient if many OLS regressions are used, but can
    lead to slightly poorer, even unstable, fits when parsimonious
    regressions are used}
  \item{validation}{ method for cross validation when applying 
    a \emph{parsimonious} regression method.  The default setting
    of \code{"CV"} (randomized 10-fold cross-validation) is the faster
    method, but does not yield a deterministic result and does not apply
    for regressions on less than ten responses.
    \code{"LOO"} (leave-one-out cross-validation)
    is deterministic, always applicable, and applied automatically whenever 
    \code{"CV"} cannot be used.  When standard least squares is
    appropriate, the methods implemented the
    \pkg{lars} package (e.g. lasso) support model choice via the
    \code{"Cp"} statistic, which defaults to the \code{"CV"} method
    when least squares fails.  This argument is ignored for the
  \code{"ridge"} method; see details below}
  \item{obs}{ logical indicating whether or not to (additionally)
    compute a mean vector and covariance matrix based only on the observed
    data, without regressions.  I.e., means are calculated as averages
    of each non-\code{NA} entry in the columns of \code{y}, and entries
    \code{(a,b)} of the
    covariance matrix are calculated by applying \code{\link{cov}(ya,yb)}
    to the jointly non-\code{NA} entries of columns \code{a} and \code{b}
    of \code{y}}
  \item{verb}{ whether or not to print progress indicators.  The default
    (\code{verb = 0}) keeps quiet, while any positive number causes brief
    statement about dimensions of each regression to print to
    the screen as it happens.  \code{verb = 2} causes each of the ML
    regression estimators to be printed along with the corresponding
    new entries of the mean and columns of the covariance matrix.
    \code{verb = 3} requires that the RETURN key be pressed between
    each print statement}
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

  Next each successive group of columns with the same missingness pattern
  is processed in sequence (assuming \code{batch = TRUE}).
  Suppose a total of \code{j} columns have
  been processed this way already.  Let \code{y2} represent the non-missing
  contingent of the next group of \code{k} columns of \code{y}
  with and identical missingness pattern, and let \code{y1} be the
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
  fails due to co-linearity then one of the other \code{method}s is tried.
  
  All \code{method}s require a scheme for estimating the amount of
  variability explained by increasing the numbers of coefficients
  (or principal components) in the model.
  Towards this end, the \pkg{pls} and \pkg{lars} packages support
  10-fold cross validation (CV) or leave-one-out (LOO) CV estimates of
  root mean squared error.  See \pkg{pls} and \pkg{lars} for
  more details.  \code{monomvn} uses
  CV in all cases except when \code{nrow(y1) <= 10}, in which case CV fails and
  LOO is used.  Whenever \code{nrow(y1) <= 3} \code{\link[pls]{pcr}}
  fails,  so \code{\link[pls]{plsr}} is used instead.
  If \code{quiet = FALSE} then a \code{\link{warning}}
  is given whenever the first choice for a regression fails.
  
  For \pkg{pls} methods, RMSEs are calculated for a number of
  components in \code{1:ncomp.max} where
  a \code{NULL} value for \code{ncomp.max} it is replaced with

  \code{ncomp.max <- min(ncomp.max, ncol(y2), nrow(y1)-1)}

  which is the max allowed by the \pkg{pls} package.
  
  Simple heuristics are used to select a small number of components
  (\code{ncomp} for \pkg{pls}), or number of coefficients (for
  \pkg{lars}), which explains a large amount of the variability (RMSE).
  The \pkg{lars} methods use a \dQuote{one-standard error rule} outlined
  in Section 7.10, page 216 of HTF below.  The
  \pkg{pls} package does not currently support the calculation of
  standard errors for CV estimates of RMSE, so a simple linear penalty
  for increasing \code{ncomp} is used instead.  The ridge constant
  (lambda) for \code{\link[MASS]{lm.ridge}} is set using the
  \code{\link{optimize}} function on the \code{GCV} output.

  Based on the ML \code{ncol(y1)+1} regression coefficients (including
  intercept) obtained for each of the
  columns of \code{y2}, and on the corresponding \code{matrix} of
  residual sum of squares, and on the previous \code{j-1} means
  and rows/cols of the covariance matrix, the \code{j:(j+k)} entries and
  rows/cols can be filled in as described by Little and Rubin, section 7.4.3.

  Once every column has been processed, the entries of the mean vector, and
  rows/cols of the covariance matrix are re-arranged into their original
  order.
}
\value{
  \code{monomvn} returns an object of class \code{"monomvn"}, which is a
  \code{list} containing a subset of the components below.

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
    \code{\link[pls]{pcr}} regression, or \code{NA} if such a method was
    not used.  This field is used to record \eqn{\lambda}{lambda}
    when \code{\link[MASS]{lm.ridge}} is used }
  \item{lambda }{if \code{method} is one of \code{c("lasso",
      "forward.stagewise", "ridge")}, then this field records the
    \eqn{\lambda}{lambda} penalty parameters used}
  \item{mu.obs }{when \code{obs = TRUE} this is the \dQuote{observed}
    mean vector}
  \item{S.obs }{when \code{obs = TRUE} this is the \dQuote{observed}
    covariance matrix, as described above.  Note that \code{S.obs} is
    usually not positive definite }
}
\references{
  Robert B. Gramacy and Joo Hee Lee (2007).
  \emph{On estimating covariances between many assets with histories 
  of highly variable length}. Preprint available on arXiv:0710.5837:\cr
  \url{http://arxiv.org/abs/0710.5837}

  Roderick J.A. Little and Donald B. Rubin (2002).
  \emph{Statistical Analysis with Missing Data}, Second Edition.
  Wilely.

  Bjorn-Helge Mevik and Ron Wehrens (2007).
  \emph{The \pkg{pls} Package: Principal Component and Partial
    Least Squares Regression in R.} 
  Journal of Statistical Software \bold{18}(2)

  Bradley Efron, Trevor Hastie, Ian Johnstone and Robert Tibshirani
  (2003).
  \emph{Least Angle Regression (with discussion).}
  Annals of Statistics \bold{32}(2); see also\cr
  \url{http://www-stat.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf}
  
  Trevor Hastie, Robert Tibshirani and Jerome Friedman (2002).
  \emph{Elements of Statistical Learning.} Springer, NY. [HTF]
  
  Some of the code for \code{monomvn}, and its subroutines, was inspired
  by code found on the world wide web, written by Daniel Heitjan,\cr
  \url{http://www.cceb.upenn.edu/pages/heitjan/courses/bsta782/examples/fcn.q}

\url{http://www.statslab.cam.ac.uk/~bobby/monomvn.html}
}
\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk}}

\note{
  The CV in \pkg{plsr} and \pkg{lars} are random in nature, and so
  can be dependent on the random seed.  Use \code{validation=LOO} for
  deterministic (but slower) result

  Be warned that the \pkg{lars} implementation of
  \code{"forward.stagewise"} can sometimes get stuck in
  (what seems like) an infinite loop.
  This is not a bug in the \code{monomvn} package;
  the bug has been reported to the authors of \pkg{lars}
}

\seealso{ \code{\link{bmonomvn}}, \code{\link[norm]{em.norm}}
  in the \pkg{norm} package, and \code{\link[mvnmle]{mlest}}
  in the \pkg{mvnmle} package}

\examples{
## standard usage, duplicating the results in
## Little and Rubin, section 7.4.3 -- try adding 
## verb=3 argument for a step-by-step breakdown
data(cement.miss)
out <- monomvn(cement.miss)
out
out$mu
out$S

##
## A bigger example, comparing the various methods
##

## generate N=100 samples from a 10-d random MVN
xmuS <- randmvn(100, 20)

## randomly impose monotone missingness
xmiss <- rmono(xmuS$x)

## plsr
oplsr <- monomvn(xmiss, obs=TRUE)
oplsr
Ellik.norm(oplsr$mu, oplsr$S, xmuS$mu, xmuS$S)

## calculate the complete and observed RMSEs
n <- nrow(xmiss) - max(oplsr$na)
x.c <- xmiss[1:n,]
mu.c <- apply(x.c, 2, mean)
S.c <- cov(x.c)*(n-1)/n
Ellik.norm(mu.c, S.c, xmuS$mu, xmuS$S)
Ellik.norm(oplsr$mu.obs, oplsr$S.obs, xmuS$mu, xmuS$S)

## plcr
opcr <- monomvn(xmiss, method="pcr")
Ellik.norm(opcr$mu, opcr$S, xmuS$mu, xmuS$S)

## ridge regression
oridge <- monomvn(xmiss, method="ridge")
Ellik.norm(oridge$mu, oridge$S, xmuS$mu, xmuS$S)

## lasso
olasso <- monomvn(xmiss, method="lasso")
Ellik.norm(olasso$mu, olasso$S, xmuS$mu, xmuS$S)

## lar
olar <- monomvn(xmiss, method="lar")
Ellik.norm(olar$mu, olar$S, xmuS$mu, xmuS$S)

## forward.stagewise
ofs <- monomvn(xmiss, method="forward.stagewise")
Ellik.norm(ofs$mu, ofs$S, xmuS$mu, xmuS$S)

## stepwise
ostep <- monomvn(xmiss, method="stepwise")
Ellik.norm(ostep$mu, ostep$S, xmuS$mu, xmuS$S)
}

\keyword{ multivariate }
\keyword{ regression }
