\name{blasso}
\alias{blasso}

\title{ Bayesian Lasso Regression }
\description{
  Ordinary least squares and Lasso regression by sampling
  from the Bayesian posterior distribution via Gibbs Sampling
  and Reversible Jump for model selection
}
\usage{
blasso(X, y, T = 100, thin = 10, RJ = TRUE, M = NULL,
       beta = NULL, lambda2 = 1, s2 = 1, tau2i = NULL,
       rd = c(2,0.1), ab = NULL, rao.s2 = TRUE,
       normalize = TRUE, verb = 1)
}

\arguments{
  \item{X}{\code{data.frame}, \code{matrix}, or vector of inputs \code{X} }
  \item{y}{ vector of output responses \code{y} of length equal to the
    leading dimension (rows) of \code{X}, i.e., \code{length(y) == nrow(X)}}
  \item{T}{ Total number of MCMC samples to be collected }
  \item{thin}{ number of MCMC samples to skip before a sample is
    collected (via thinning) }
  \item{RJ}{ if \code{TRUE} then model selection on the columns of the
    design matrix (and thus the parameter \code{beta} in the model) is
    performed by reversible jump (RJ) MCMC.  The initial model is
    specified by the \code{beta} input, described below, and the maximal
    number of covariates in the model is specified by \code{M} }
  \item{M}{ the maximal number of allowed covariates (columns of
    \code{X}) in the model.  If input \code{lambda2 > 0} then \code{M}
    as large as \code{ncol(X)} is allowed.  Otherwise \code{M} must be
    \code{<= min(ncol(X), length(y)-1)}, its default value when argument
    \code{NULL} is given }
  \item{beta}{ Initial setting of the regression coefficients.  Any
    zero-components will imply the corresponding covariate (column
    of \code{X}) is not in the initial model.  When input \code{RJ =
      FALSE} (no RJ) and \code{lambda2 > 0} (use lasso) then no
    components are allowed to be exactly zero.  The default setting is
    therefore contextual.  See below for details }
  \item{lambda2}{ square of the initial lasso penalty parameter.  If
    zero, the least squares regressions are used }
  \item{s2}{ initial variance parameter }
  \item{tau2i}{ initial vector of lasso latent-variables along the diagonal
    of the covariance matrix in the prior for beta.  The default setting
    (when \code{NULL} is given) is contextual.  See below for details }
  \item{rd}{ \code{=c(r, delta)}, the alpha (shape) parameter and beta
    (rate) parameter to the gamma distribution prior for
   the lasso parameter lambda }
 \item{ab}{ \code{=c(a, b)}, the alpha (shape) parameter and the beta
   (scale) parameter for the inverse-gamma distribution
   prior for the variance parameter \code{s2}.  A default of \code{NULL}
   generates appropriate non-informative values depending on the
   nature of the regression }
  \item{rao.s2 }{indicates whether to use Rao-Blackwellized samples for
    \eqn{\sigma^2}{s^2} should be used (default \code{TRUE}), see
    the details section, below}
  \item{normalize}{ if \code{TRUE}, each variable is standardized to have unit
    L2 norm, otherwise it is left alone. Default is \code{TRUE} }
  \item{verb}{ verbosity level; currently only \code{verb = 0} and
    \code{verb = 1} are supported }
}
\details{
  The Bayesian lasso model and Gibbs Sampling algorithm is described
  in detail in Park \& Casella (2008).  The algorithm implemented
  by this function is identical to that described therein., with
  the exception of the \dQuote{option} to use a Rao-Blackwellized sample
  of \eqn{\sigma^2}{s^2} (with \eqn{\beta}{beta} integrated out)
  for improved mixing, and the model selections by RJ described below.
  When input argument \code{lambda2 = 0} is
  supplied, the model is a simple hierarchical linear model where
  the prior for \eqn{\beta}{beta} has mean zero and diagonal
  covariance matrix with diagonal \code{1/tau2i}

  Specifying \code{RJ = TRUE} causes Bayesian model selection and
  averaging to commence for choosing which of the columns of the
  design matrix \code{X} (and thus parameters \code{beta}) are
  included in the model.  The zero-components of the initial \code{beta}
  vector specify which of the columns are in the initial model, and
  \code{M} specifies the maximal number of columns.

  The RJ mechanism implemented here is distinct from the
  model selection method described by Hans (2008), which is based on
  Geweke (1996).  Those methods require departing from the Park \& Casella
  (2008) to sample sampling from each conditional
  \eqn{\beta_i | \beta_{(-i)}, \dots}{beta[i] | beta[-i], ...} for all
  \eqn{i}{i}, and that a mixture prior with a point-mass at zero be
  placed on each \eqn{\beta_i}{beta[i]}.  The method implemented
  here requires no such special prior and retains the joint sampling
  from the full \eqn{\beta}{beta} vector of non-zero entries, which
  we believe yields better mixing in the Markov chain.  RJ
  proposals to increase/decrease the number of non-zero entries
  does proceed component-wise, but the acceptance rates are high due
  to an optimal second--order proposal (Ehlers \& Brooks, 2005)
}
\value{
  \code{blasso} returns an object of class \code{"blasso"}, which is a
  list containing a copy of all of the input arguments as well as
  of the components listed below.

  \item{call }{a copy of the function call as used}
  \item{mu }{ a vector of \code{T} samples from the (un-penalized)
    \dQuote{intercept} parameter }
  \item{beta }{ a \code{T*ncol(X)} \code{matrix} of \code{T} samples from
    the (penalized) regression coefficients}
  \item{m }{ the number of non-zero entries in each vector of \code{T}
    samples of \code{beta}}
  \item{s2 }{ a vector of \code{T} samples from the variance parameter}
  \item{lambda2 }{ a vector of \code{T} samples from the penalty
    parameter}
  \item{tau2i}{ a \code{T*ncol(X)} \code{matrix} of \code{T} samples from
    the (latent) inverse diagonal of the prior covariance matrix for
    \code{beta}}
}
\note{
  Whenever \code{ncol(X) >= nrow(X)} it must be that either \code{RJ = TRUE}
  with \code{M <= nrow(X)-1} (the default) or that the lasso is turned
  on with \code{lambda2 > 0} or the regression problem is ill-posed.

  When \code{lambda2 = 0} is given the initial \code{tau2i} vector is
  taken to be zero to indicate that it has been removed from the model,
  implying a flat, improper, prior for \code{beta}.
  Other settings will be taken, but are ignored in the current version.
  Future versions will allow the specification of informative
  (non lasso) priors.
  
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
   \code{\link{regress}}
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
reg.blas <- blasso(x, y, T=1000)

## summarize the beta (regression coefficients) estimates
plot(reg.blas, burnin=200)
points(drop(reg.las$b), col=2, pch=20)
points(drop(reg.ols$b), col=3, pch=18)
legend("topleft", c("lasso", "lsr"), col=2:3, pch=c(20,18))

## plot the size of different models visited
plot(reg.blas, burnin=200, which="m")

## (TODO: allow Bayes method with fixed lambda)

## get the summary
s <- summary(reg.blas)

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

xmuS <- randmvn(50, 101)
X <- xmuS$x[,1:100]
y <- drop(xmuS$x[,101])
out.b <- blasso(X, y, T=1000, thin=20, verb=0)

## plot summary of the model order
plot(out.b, burnin=100, which="m")

## fit a standard lasso model
out.las <- regress(X, y, method="lasso")

## compare via RMSE
beta <- xmuS$S[101,-101] %*% solve(xmuS$S[-101,-101])
sqrt(mean((apply(out.b$beta, 2, mean) - beta)^2))
sqrt(mean((out.las$b[-1] - beta)^2))

}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ regression }
