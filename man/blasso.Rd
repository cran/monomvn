\name{blasso}
\alias{blasso}
\alias{bridge}

\title{ Bayesian Lasso and Ridge Regression }
\description{
  Inference for ordinary least squares, lasso and ridge regression
  models by (Gibbs) sampling from the Bayesian posterior distribution,
  augmented with Reversible Jump for model selection
}
\usage{
bridge(X, y, T = 1000, thin = NULL, RJ = TRUE, M = NULL,
       beta = NULL, lambda2 = 1, s2 = 1, rd = NULL,
       ab = NULL, rao.s2 = TRUE, normalize = TRUE,
       verb = 1)
blasso(X, y, T = 1000, thin = NULL, RJ = TRUE, M = NULL,
       beta = NULL, lambda2 = 1, s2 = 1, ridge = FALSE,
       rd = NULL, ab = NULL, rao.s2 = TRUE,
       normalize = TRUE, verb = 1)
}

\arguments{
  \item{X}{\code{data.frame}, \code{matrix}, or vector of inputs \code{X} }
  \item{y}{ vector of output responses \code{y} of length equal to the
    leading dimension (rows) of \code{X}, i.e., \code{length(y) == nrow(X)}}
  \item{T}{ total number of MCMC samples to be collected }
  \item{thin}{ number of MCMC samples to skip before a sample is
    collected (via thinning).  If \code{NULL} (default), then
    \code{thin} is determined based on the regression model implied
    by \code{RJ}, \code{lambda2}, and \code{ncol(X)} }
  \item{RJ}{ if \code{TRUE} then model selection on the columns of the
    design matrix (and thus the parameter \code{beta} in the model) is
    performed by Reversible Jump (RJ) MCMC.  The initial model is
    specified by the \code{beta} input, described below, and the maximal
    number of covariates in the model is specified by \code{M} }
  \item{M}{ the maximal number of allowed covariates (columns of
    \code{X}) in the model.  If input \code{lambda2 > 0} then
    \code{M <= ncol(X)} is allowed.  Otherwise
    \code{M <= min(ncol(X), length(y)-1)}, which is default value
    when a \code{NULL} argument is given }
  \item{beta}{ initial setting of the regression coefficients.  Any
    zero-components will imply that the corresponding covariate (column
    of \code{X}) is not in the initial model.  When input \code{RJ =
      FALSE} (no RJ) and \code{lambda2 > 0} (use lasso) then no
    components are allowed to be exactly zero.  The default setting is
    therefore contextual.  See below for details }
  \item{lambda2}{ square of the initial lasso penalty parameter.  If
    zero, then least squares regressions are used }
  \item{s2}{ initial variance parameter }
  \item{ridge}{ specifies if ridge regression should be done instead
    of the lasso; only meaningful when \code{lambda2 > 0} }
  \item{rd}{ \code{=c(r, delta)}, the alpha (shape) parameter and beta
    (rate) parameter to the gamma distribution prior for
    the \eqn{\lambda^2}{lambda2} parameter under the lasso model;
    or, the alpha (shape) parameter and beta (scale) parameter to the
    inverse-gamma distribution prior for the \eqn{\lambda^2}{lambda2}
    parameter under the ridge regression model. A default of \code{NULL}
   generates appropriate non-informative values depending on the
   nature of the regression.  See the details below for information
   on the special settings of this paper for ridge regression }
 \item{ab}{ \code{=c(a, b)}, the alpha (shape) parameter and the beta
   (scale) parameter for the inverse-gamma distribution
   prior for the variance parameter \code{s2}.  A default of \code{NULL}
   generates appropriate non-informative values depending on the
   nature of the regression }
  \item{rao.s2 }{indicates whether to use Rao-Blackwellized samples for
    \eqn{\sigma^2}{s^2} should be used (default \code{TRUE}), see
    the details section, below}
  \item{normalize}{ if \code{TRUE}, each variable is standardized to have unit
    L2-norm, otherwise it is left alone; default is \code{TRUE} }
  \item{verb}{ verbosity level; currently only \code{verb = 0} and
    \code{verb = 1} are supported }
}
\details{
  The Bayesian lasso model and Gibbs Sampling algorithm is described
  in detail in Park \& Casella (2008).  The algorithm implemented
  by this function is identical to that described therein, with
  the exception of an added \dQuote{option} to use a Rao-Blackwellized sample
  of \eqn{\sigma^2}{s^2} (with \eqn{\beta}{beta} integrated out)
  for improved mixing, and the model selections by RJ described below.
  When input argument \code{lambda2 = 0} is
  supplied, the model is a simple hierarchical linear model where
  \eqn{(\beta,\sigma^2)}{(beta,s2)} is given an Jeffrey's prior

  Specifying \code{RJ = TRUE} causes Bayesian model selection and
  averaging to commence for choosing which of the columns of the
  design matrix \code{X} (and thus parameters \code{beta}) should be
  included in the model.  The zero-components of the \code{beta} input
  specify which columns are in the initial model, and
  \code{M} specifies the maximal number of columns.

  The RJ mechanism distinguishes implementation of the Bayesian lasso
  model selection from the one described by Hans (2008),
  which is based on an idea from Geweke (1996).
  Those methods require departing from the Park \& Casella
  (2008) latent-variable model and requires sampling from each conditional
  \eqn{\beta_i | \beta_{(-i)}, \dots}{beta[i] | beta[-i], ...} for all
  \eqn{i}{i}, since a mixture prior with a point-mass at zero is
  placed on each \eqn{\beta_i}{beta[i]}.  The method implemented
  here requires no such special prior and retains the joint sampling
  from the full \eqn{\beta}{beta} vector of non-zero entries, which
  we believe yields better mixing in the Markov chain.  RJ
  proposals to increase/decrease the number of non-zero entries
  does proceed component-wise, but the acceptance rates are high due
  to an optimal second--order proposal (Ehlers \& Brooks, 2005).

  Bayesian ridge regression is implemented as a special case via the
  \code{bridge} function.  This essentially calls \code{blasso}
  with \code{ridge = TRUE} and \code{RJ = FALSE} (by default).
  A default setting of \code{rd = c(0,0)} is implied by \code{rd =
    NULL}, giving the Jeffery's prior for the penalty parameter
  \eqn{\lambda^2}{lambda^2} unless \code{ncol(X) >= length(y)} in
  which case the proper specification of \code{rd = c(5,10)} is used
  instead
}
\value{
  \code{blasso} returns an object of class \code{"blasso"}, which is a
  list containing a copy of all of the input arguments as well as
  of the components listed below.

  \item{call }{a copy of the function call as used}
  \item{mu }{ a vector of \code{T} samples of the (un-penalized)
    \dQuote{intercept} parameter }
  \item{beta }{ a \code{T*ncol(X)} \code{matrix} of \code{T} samples from
    the (penalized) regression coefficients}
  \item{m }{ the number of non-zero entries in each vector of \code{T}
    samples of \code{beta}}
  \item{s2 }{ a vector of \code{T} samples of the variance parameter}
  \item{lambda2 }{ a vector of \code{T} samples of the penalty
    parameter}
  \item{tau2i}{ a \code{T*ncol(X)} \code{matrix} of \code{T} samples from
    the (latent) inverse diagonal of the prior covariance matrix for
    \code{beta}}
}
\note{
  Whenever \code{ncol(X) >= nrow(X)} it must be that either \code{RJ = TRUE}
  with \code{M <= nrow(X)-1} (the default) or that the lasso is turned
  on with \code{lambda2 > 0}.  Otherwise the regression problem is ill-posed.

  Since the starting values are considered to be first sample (of
  \code{T}), the total number of (new) samples obtained by Gibbs
  Sampling will be \code{T-1}
}

\references{
  Park, T., Casella, G. (2008).
  \emph{The Bayesian Lasso}, (unpublished)\cr
  \url{http://www.stat.ufl.edu/~casella/Papers/bayeslasso.pdf}

  Chris Hans. (2008). \emph{Bayesian Lasso regression.}
  Technical Report No. 810, Department of Statistics,
  The Ohio State University, Columbus, OH 43210.\cr
  \url{http://www.stat.osu.edu/~hans/Papers/blasso.pdf}

  Geweke, J. (1996). \emph{Variable selection and model comparison
    in regression.} In Bayesian Statistics 5.  Editors: J.M. Bernardo,
  J.O. Berger, A.P. Dawid and A.F.M. Smith, 609-620. Oxford Press.
  
  Ehlers, R.S. and Brooks, S.P. (2005).
  \emph{Efficient Construction of Reversible Jump MCMC Proposals for
    Autoregressive Time Series Models.} (unpublished)\cr
  \url{http://www.statslab.cam.ac.uk/~steve/mypapers/ehlb02.ps}
  
  \url{http://www.statslab.cam.ac.uk/~bobby/monomvn.html}
}

\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk} }

 \seealso{
   \code{\link{lm}} ,
   \code{\link[lars]{lars}} in the \pkg{lars} package,
   \code{\link{regress}},
   \code{\link[MASS]{lm.ridge}} in the \pkg{MASS} package
 }

 \examples{
## following the lars diabetes example
data(diabetes)
attach(diabetes)

## Ordinary Least Squares regression
reg.ols <- regress(x, y)

## Lasso regression
reg.las <- regress(x, y, method="lasso")

## Bayesian Lasso regression
reg.blas <- blasso(x, y)

## summarize the beta (regression coefficients) estimates
plot(reg.blas, burnin=200)
points(drop(reg.las$b), col=2, pch=20)
points(drop(reg.ols$b), col=3, pch=18)
legend("topleft", c("lasso", "lsr"), col=2:3, pch=c(20,18))

## plot the size of different models visited
plot(reg.blas, burnin=200, which="m")

## get the summary
s <- summary(reg.blas, burnin=200)

## calculate the probability that each beta coef != zero
s$bn0

## summarize s2
plot(reg.blas, burnin=200, which="s2")
s$s2

## summarize lambda2
plot(reg.blas, burnin=200, which="lambda2")
s$lambda2

## clean up
detach(diabetes)

##
## a big-p small-n example
##

n <- 25; m <- 51
xmuS <- randmvn(n, m)
X <- xmuS$x[,1:(m-1)]
Y <- drop(xmuS$x[,m])
obl <- blasso(X, Y, verb=0)

## plot summary of the model order
plot(obl, burnin=10, which="m")

## fit a standard lasso model
oml <- regress(X, Y, method="lasso")

## compare via RMSE, most often blasso will win
beta <- xmuS$S[m,-m] %*% solve(xmuS$S[-m,-m])
sqrt(mean((apply(obl$beta, 2, mean) - beta)^2))
sqrt(mean((oml$b[-1] - beta)^2))

## now try both Bayesian & ML ridge regression
obr <- bridge(X, Y, verb=0)
omr <- regress(X, Y, method="ridge")
sqrt(mean((omr$b[-1] - beta)^2))
sqrt(mean((apply(obr$beta[-c(1:200),], 2, mean) - beta)^2))
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ regression }
