\name{blasso}
\alias{blasso}

\title{ Bayesian Lasso Regression }
\description{
  Ordinary least squares and Lasso regression by sampling
  from the Bayesian posterior distribution via Gibbs Sampling
}
\usage{
blasso(X, y, T = 100, thin = 10, lambda2 = 1, s2 = 1,
       tau2i = rep(1, ncol(X)), r = 1, delta = 1, a = 0, b = 0,
       rao.s2 = TRUE, normalize = TRUE, verb = 1)
}

\arguments{
  \item{X}{\code{data.frame}, \code{matrix}, or vector of inputs \code{X} }
  \item{y}{ vector of output responses \code{y} of length equal to the
    leading dimension (rows) of \code{X}, i.e., \code{length(y) == nrow(X)}}
  \item{T}{ Total number of MCMC samples to be collected }
  \item{thin}{ number of MCMC samples to skip before a sample is
    collected (via thinning) }
  \item{lambda2}{ square of the initial lasso penalty parameter.  If
    zero, the least squares regressions are used }
  \item{s2}{ initial variance parameter }
  \item{tau2i}{ initial vector of lasso latent-variables along the diagonal
    of the covariance matrix in the prior for beta }
 \item{r}{ alpha (shape) parameter to the gamma distribution prior for
   the lasso parameter lambda }
  \item{delta}{ beta (rate) parameter to the gamma distribution prior
    for the lasso parameter lambda }
  \item{a}{ alpha (shape) parameter for the inverse-gamma distribution
    prior for the variance parameter \code{s2} }
  \item{b}{ beta (scale) parameter for the inverse-gamma distribution
    prior for the variance parameter \code{s2} }
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
  for improved mixing.  When input argument \code{lambda2 = 0} is
  supplied, the model is a simple hierarchical linear model where
  the prior for \eqn{\beta}{beta} has mean zero and diagonal
  covariance matrix with diagonal \code{1/tau2i}
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
  \item{s2 }{ a vector of \code{T} samples from the variance parameter}
  \item{lambda2 }{ a vector of \code{T} samples from the penalty
    parameter}
  \item{tau2i }{ a \code{T*ncol(X)} \code{matrix} of \code{T} samples from
    the (latent) inverse diagonal of the prior covariance matrix for
    \code{beta}}
}
\note{
  Since the starting values are considered to be frist sample (of
  \code{T}), the total number of (new) samples obtained by Gibbs
  Sampling will be \code{T-1}
  }
\references{
  Park, T., Casella, G. (2008).
  \emph{The Bayesian Lasso}, (unpublished)
  \url{http://www.stat.ufl.edu/~casella/Papers/bayeslasso.pdf}

\url{http://www.statslab.cam.ac.uk/~bobby/monomvn.html}
}

\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk} }

%\note{ ~~further notes~~ 
 %~Make other sections like Warning with \section{Warning }{....} ~
 %}
 
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
points(drop(reg.las$b), col=2, pch=19)
points(drop(reg.ols$b), col=3, pch=18)
abline(h=0, lty=2, lwd=2)
legend("topleft", c("lasso", "lsr"), col=2:3, pch=19:18)

## summarize s2
plot(reg.blas, burnin=200, which="s2")

## summarize the posterior distribution of lambda2 and s2
detach(diabetes)
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ regression }
