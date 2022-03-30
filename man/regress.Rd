\name{regress}
\alias{regress}

\title{ Switch function for least squares and parsimonious monomvn regressions }
\description{
  This function fits the specified ordinary least squares or
  parsimonious regression (plsr, pcr, ridge, and lars methods)
  depending on the arguments provided, and returns estimates of
  coefficients and (co-)variances in a \code{monomvn} friendly
  format
}
\usage{
regress(X, y, method = c("lsr", "plsr", "pcr", "lasso", "lar",
     "forward.stagewise", "stepwise", "ridge", "factor"), p = 0,
     ncomp.max = Inf, validation = c("CV", "LOO", "Cp"),
     verb = 0, quiet = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{\code{data.frame}, \code{matrix}, or vector of inputs \code{X} }
  \item{y}{ matrix of responses \code{y} of row-length equal to the
    leading dimension (rows) of \code{X}, i.e., \code{nrow(y) ==
      nrow(X)}; if \code{y} is a vector, then \code{nrow} may be
    interpreted as \code{length} }
  \item{method}{ describes the type of \emph{parsimonious}
    (or \emph{shrinkage}) regression, or ordinary least squares.
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
    columns of \code{y} as known factors}
  \item{p}{ when performing regressions, \code{0 <= p <= 1}
    is the proportion of the
    number of columns to rows in the design matrix before an
    alternative regression \code{method} (except \code{"lsr"})
    is performed as if  least-squares regression \dQuote{failed}.
    Least-squares regression is
    known to fail when the number of columns is greater than or
    equal to the number of rows.
    The default setting, \code{p = 0}, forces the specified
    \code{method} to be used for \emph{every} regression unless
    \code{method = "lsr"} is specified but is unstable.
    Intermediate settings of \code{p} allow the user
    to specify that least squares regressions are preferred only
    when there are sufficiently more rows in the design matrix
    (\code{X}) than columns. When \code{method = "factor"} the \code{p}
    argument represents an integer (positive) number of initial columns
    of \code{y} to treat as known factors}
  \item{ncomp.max}{ maximal number of (principal) components to consider
    in a \code{method}---only meaningful for the \code{"plsr"} or
    \code{"pcr"} methods.  Large settings can cause the execution to be
    slow as they drastically increase the cross-validation (CV) time}
  \item{validation}{ method for cross validation when applying 
    a \emph{parsimonious} regression method.  The default setting
    of \code{"CV"} (randomized 10-fold cross-validation) is the faster method, 
    but does not yield a deterministic result and does not apply for
    regressions on less than ten responses. \code{"LOO"}
    (leave-one-out cross-validation)
    is deterministic, always applicable, and applied automatically whenever 
    \code{"CV"} cannot be used.  When standard least squares is
  appropriate, the methods implemented the
  \pkg{lars} package (e.g. lasso) support model choice via the
  \code{"Cp"} statistic, which defaults to the \code{"CV"} method
  when least squares fails.  This argument is ignored for the
  \code{"ridge"} method; see details below}
  \item{verb}{ whether or not to print progress indicators.  The default
    (\code{verb = 0}) keeps quiet.  This argument is provided for
    \code{\link{monomvn}} and is not intended to be set by the user
    via this interface }
  \item{quiet}{ causes \code{\link{warning}}s about regressions to be silenced
    when \code{TRUE}}
}
\details{
  All \code{method}s (except \code{"lsr"}) require a scheme for
  estimating the amount of variability explained by increasing numbers
  of non-zero coefficients (or principal components) in the model.
  Towards this end, the \pkg{pls} and \pkg{lars} packages support
  10-fold cross validation (CV) or leave-one-out (LOO) CV estimates of
  root mean squared error.  See \pkg{pls} and \pkg{lars} for
  more details.  The \code{regress} function uses CV in all cases
  except when \code{nrow(X) <= 10}, in which case CV fails and
  LOO is used.  Whenever \code{nrow(X) <= 3} \code{\link[pls]{pcr}}
  fails,  so \code{\link[pls]{plsr}} is used instead.
  If \code{quiet = FALSE} then a \code{\link{warning}}
  is given whenever the first choice for a regression fails.
  
  For \pkg{pls} methods, RMSEs
  are calculated for a number of components in \code{1:ncomp.max} where
  a \code{NULL} value for \code{ncomp.max} it is replaced with

  \code{ncomp.max <- min(ncomp.max, ncol(y), nrow(X)-1)}

  which is the max allowed by the \pkg{pls} package.
  
  Simple heuristics are used to select a small number of components
  (\code{ncomp} for \pkg{pls}), or number of coefficients (for
  \pkg{lars}) which explains a large amount of the variability (RMSE).
  The \pkg{lars} methods use a \dQuote{one-standard error rule} outlined
  in Section 7.10, page 216 of HTF below.  The
  \pkg{pls} package does not currently support the calculation of
  standard errors for CV estimates of RMSE, so a simple linear penalty
  for increasing \code{ncomp} is used instead.  The ridge constant
  (lambda) for \code{\link[MASS]{lm.ridge}} is set using the \code{\link{optimize}}
  function on the \code{GCV} output.
}
\value{
  \code{regress} returns a \code{list} containing
  the components listed below.

  \item{call }{a copy of the function call as used}
  \item{method }{a copy of the \code{method} input argument}
  \item{ncomp }{depends on the \code{method} used: is \code{NA} when
    \code{method = "lsr"}; is the number of principal
    components for \code{method = "pcr"} and \code{method = "plsr"};
    is the number of non-zero components in the coefficient vector
    (\code{$b}, not counting the intercept) for any of the
    \code{\link[lars]{lars}} methods; and gives the chosen
    \eqn{\lambda}{lambda} penalty parameter for \code{method = "ridge"}}
  \item{lambda }{if \code{method} is one of \code{c("lasso",
      "forward.stagewise", "ridge")}, then this field records the
    \eqn{\lambda}{lambda} penalty parameter used}
  \item{b }{matrix containing the estimated regression coefficients,
    with \code{ncol(b) = ncol(y)} and the intercept
    in the first row}
  \item{S }{(biased corrected) maximum likelihood estimate of residual
    covariance matrix}
}

\references{
  Bjorn-Helge Mevik and Ron Wehrens (2007).
  \emph{The \pkg{pls} Package: Principal Component and Partial
    Least Squares Regression in R.} 
  Journal of Statistical Software \bold{18}(2)

  Bradley Efron, Trevor Hastie, Ian Johnstone and Robert Tibshirani
  (2003).
  \emph{Least Angle Regression (with discussion).}
  Annals of Statistics \bold{32}(2); see also \cr
  \url{https://hastie.su.domains/Papers/LARS/LeastAngle_2002.pdf}
  
\url{https://bobby.gramacy.com/r_packages/monomvn/}
}

\author{ Robert B. Gramacy \email{rbg@vt.edu} }

\note{
  The CV in \pkg{plsr} and \pkg{lars} are random in nature, and so
  can be dependent on the random seed.  Use \code{validation="LOO"} for
  deterministic (but slower) result

  Be warned that the \pkg{lars} implementation of
  \code{"forward.stagewise"} can sometimes get stuck in
  (what seems like) an infinite loop.
  This is not a bug in the \code{regress} function;
  the bug has been reported to the authors of \pkg{lars}
}
\seealso{ \code{\link{monomvn}}, \code{\link{blasso}},
  \code{\link[lars]{lars}} in the \pkg{lars} library,
  \code{\link[MASS]{lm.ridge}} in the \pkg{MASS} library,
  \code{\link[pls]{plsr}} and \code{\link[pls]{pcr}} in the
  \pkg{pls} library
}
\examples{
## following the lars diabetes example
data(diabetes)
attach(diabetes)

## Ordinary Least Squares regression
reg.ols <- regress(x, y)

## Lasso regression
reg.lasso <- regress(x, y, method="lasso")

## partial least squares regression
reg.plsr <- regress(x, y, method="plsr")

## ridge regression
reg.ridge <- regress(x, y, method="ridge")

## compare the coefs
data.frame(ols=reg.ols$b, lasso=reg.lasso$b,
           plsr=reg.plsr$b, ridge=reg.ridge$b)

## summarize the posterior distribution of lambda2 and s2
detach(diabetes)
}

\keyword{ regression }
