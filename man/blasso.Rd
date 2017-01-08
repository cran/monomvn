\name{blasso}
\alias{blasso}
\alias{bhs}
\alias{bridge}

\title{ Bayesian Lasso/NG, Horseshoe, and Ridge Regression }
\description{
  Inference for ordinary least squares, lasso/NG, horseshoe and ridge 
  regression models by (Gibbs) sampling from the Bayesian posterior 
  distribution, augmented with Reversible Jump for model selection
}
\usage{
bhs(X, y, T=1000, thin=NULL, RJ=TRUE, M=NULL, beta=NULL,
         lambda2=1, s2=var(y-mean(y)), mprior=0, ab=NULL,
         theta=0, rao.s2=TRUE, icept=TRUE, normalize=TRUE, verb=1)
bridge(X, y, T = 1000, thin = NULL, RJ = TRUE, M = NULL,
       beta = NULL, lambda2 = 1, s2 = var(y-mean(y)), mprior = 0,
       rd = NULL, ab = NULL, theta=0, rao.s2 = TRUE, icept = TRUE,
       normalize = TRUE, verb = 1)
blasso(X, y, T = 1000, thin = NULL, RJ = TRUE, M = NULL,
       beta = NULL, lambda2 = 1, s2 = var(y-mean(y)),
       case = c("default", "ridge", "hs", "ng"), mprior = 0, rd = NULL,
       ab = NULL, theta = 0, rao.s2 = TRUE, icept = TRUE, 
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
    by \code{RJ}, \code{lambda2}, and \code{ncol(X)}; and also
    on the errors model implied by \code{theta} and \code{nrow(X)} }
  \item{RJ}{ if \code{TRUE} then model selection on the columns of the
    design matrix (and thus the parameter \code{beta} in the model) is
    performed by Reversible Jump (RJ) MCMC.  The initial model is
    specified by the \code{beta} input, described below, and the maximal
    number of covariates in the model is specified by \code{M} }
  \item{M}{ the maximal number of allowed covariates (columns of
    \code{X}) in the model.  If input \code{lambda2 > 0} then any
    \code{M <= ncol(X)} is allowed.  Otherwise it must be that
    \code{M <= min(ncol(X), length(y)-1)}, which is default value
    when a \code{NULL} argument is given }
  \item{beta}{ initial setting of the regression coefficients.  Any
    zero-components will imply that the corresponding covariate (column
    of \code{X}) is not in the initial model.  When input \code{RJ =
      FALSE} (no RJ) and \code{lambda2 > 0} (use lasso) then no
    components are allowed to be exactly zero.  The default setting is
    therefore contextual; see below for details }
  \item{lambda2}{ square of the initial lasso penalty parameter.  If
    zero, then least squares regressions are used }
  \item{s2}{ initial variance parameter }
  \item{case}{ specifies if ridge regression, the
    Normal-Gamma, or the horseshoe prior should be done instead
    of the lasso; only meaningful when \code{lambda2 > 0} }
  \item{mprior}{ prior on the number of non-zero regression coefficients
    (and therefore covariates) \code{m} in the model. The default
    (\code{mprior = 0}) encodes the uniform prior on \code{0 <= m <= M}.
    A scalar value \code{0 < mprior < 1} implies a Binomial prior
    \code{Bin(m|n=M,p=mprior)}. A 2-vector \code{mprior=c(g,h)}
    of positive values \code{g} and \code{h} represents
    gives \code{Bin(m|n=M,p)} prior where \code{p~Beta(g,h)} }
  \item{rd}{ \code{=c(r, delta)}, the alpha (shape) parameter and
    \eqn{\beta}{beta} (rate) parameter to the gamma distribution prior
    \code{G(r,delta)} for the \eqn{\lambda^2}{lambda2} parameter under
    the lasso model; or, the \eqn{\alpha}{alpha} (shape) parameter and
    \eqn{\beta}{beta} (scale) parameter to the
    inverse-gamma distribution \code{IG(r/2, delta/2)} prior for
    the \eqn{\lambda^2}{lambda2}
    parameter under the ridge regression model. A default of \code{NULL}
   generates appropriate non-informative values depending on the
   nature of the regression.  Specifying \code{rd=FALSE} causes
  \code{lambda2} values to be fixed at their starting value, i.e., not
  sampled.  See the details below for information
   on the special settings for ridge regression }
 \item{ab}{ \code{=c(a, b)}, the \eqn{\alpha}{alpha} (shape)
   parameter and the \eqn{\beta}{beta} (scale) parameter for the
   inverse-gamma distribution prior \code{IG(a,b)} for the variance
   parameter \code{s2}.  A default of \code{NULL} generates appropriate
   non-informative values depending on the nature of the regression }
 \item{theta}{ the rate parameter (\code{> 0}) to the exponential prior
   on the degrees of freedom paramter \code{nu} under a model with
   Student-t errors implemented by a scale-mixture prior.
   The default setting of \code{theta = 0} turns off this prior,
   defaulting to a normal errors prior }
  \item{rao.s2}{indicates whether Rao-Blackwellized samples for
    \eqn{\sigma^2}{s^2} should be used (default \code{TRUE}); see
    below for more details }
  \item{icept}{ if \code{TRUE}, an implicit intercept term is fit
    in the model, otherwise the the intercept is zero; default is
    \code{TRUE} }
  \item{normalize}{ if \code{TRUE}, each variable is standardized
    to have unit L2-norm, otherwise it is left alone; default is
    \code{TRUE} }
  \item{verb}{ verbosity level; currently only \code{verb = 0} and
    \code{verb = 1} are supported }
}
\details{
  The Bayesian lasso model and Gibbs Sampling algorithm is described
  in detail in Park & Casella (2008).  The algorithm implemented
  by this function is identical to that described therein, with
  the exception of an added \dQuote{option} to use a Rao-Blackwellized
  sample of \eqn{\sigma^2}{s^2} (with \eqn{\beta}{beta} integrated out)
  for improved mixing, and the model selections by RJ described below.
  When input argument \code{lambda2 = 0} is
  supplied, the model is a simple hierarchical linear model where
  \eqn{(\beta,\sigma^2)}{(beta,s2)} is given a Jeffrey's prior

  Specifying \code{RJ = TRUE} causes Bayesian model selection and
  averaging to commence for choosing which of the columns of the
  design matrix \code{X} (and thus parameters \code{beta}) should be
  included in the model.  The zero-components of the \code{beta} input
  specify which columns are in the initial model, and
  \code{M} specifies the maximal number of columns.

  The RJ mechanism implemented here for the Bayesian lasso model
  selection differs from the one described by Hans (2009),
  which is based on an idea from Geweke (1996).
  Those methods require departing from the Park & Casella
  (2008) latent-variable model and requires sampling from each conditional
  \eqn{\beta_i | \beta_{(-i)}, \dots}{beta[i] | beta[-i], ...} for all
  \eqn{i}{i}, since a mixture prior with a point-mass at zero is
  placed on each \eqn{\beta_i}{beta[i]}.  Out implementation
  here requires no such special prior and retains the joint sampling
  from the full \eqn{\beta}{beta} vector of non-zero entries, which
  we believe yields better mixing in the Markov chain.  RJ
  proposals to increase/decrease the number of non-zero entries
  does proceed component-wise, but the acceptance rates are high due
  due to marginalized between-model moves (Troughton & Godsill, 1997).

  When the lasso prior or RJ is used, the automatic thinning level
  (unless \code{thin != NULL}) is determined by the number of columns
  of \code{X} since this many latent variables are introduced

  Bayesian ridge regression is implemented as a special case via the
  \code{bridge} function.  This essentially calls \code{blasso}
  with \code{case = "ridge"}. A default setting of \code{rd = c(0,0)} is
  implied by \code{rd = NULL}, giving the Jeffery's prior for the
  penalty parameter \eqn{\lambda^2}{lambda^2} unless \code{ncol(X) >=
    length(y)} in which case the proper specification of \code{rd =
    c(5,10)} is used instead.

  The Normal--Gamma prior (Griffin & Brown, 2009) is implemented as
  an extension to the Bayesian lasso with \code{case = "ng"}.  Many
  thanks to James Scott for providing the code needed to extend the
  method(s) to use the horseshoe prior (Carvalho, Polson, Scott, 2010).

  When \code{theta > 0} then the Student-t errors via scale mixtures
  (and thereby extra latent variables \code{omega2}) of Geweke (1993)
  is applied as an extension to the Bayesian lasso/ridge model.
  If Student-t errors are used the automatic thinning level
  is augmented (unless \code{thin != NULL}) by the number of rows
  in \code{X} since this many latent variables are introduced
}
\value{

  \code{blasso} returns an object of class \code{"blasso"}, which is a
  \code{list} containing a copy of all of the input arguments as well as
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
  \item{gamma }{ a vector of \code{T} with the gamma parameter
    when \code{case = "ng"} }
  \item{tau2i}{ a \code{T*ncol(X)} \code{matrix} of \code{T} samples from
    the (latent) inverse diagonal of the prior covariance matrix for
    \code{beta}, obtained for Lasso regressions }
  \item{omega2}{ a \code{T*nrow(X)} \code{matrix} of \code{T} samples
    from the (latent) diagonal of the covariance matrix of the response
    providing a scale-mixture implementation of Student-t errors with
    degrees of freedom \code{nu} when active (input \code{theta > 0}) }
  \item{nu}{ a vector of \code{T} samples of the degrees of freedom
    parameter to the Student-t errors mode when active
    (input \code{theta > 0}) }
  \item{pi}{ a vector of \code{T} samples of the Binomial proportion
    \code{p} that was given a Beta prior, as described above for the
    2-vector version of the \code{mprior} input}
  \item{lpost}{ the log posterior probability of each (saved) sample of the
    joint parameters }
  \item{llik}{ the log likelihood of each (saved) sample of the
    parameters }
  \item{llik.norm}{ the log likelihood of each (saved) sample of the
    parameters under the Normal errors model when sampling under the
    Student-t model; i.e., it is not present
    unless \code{theta > 0} }
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
  Park, T., Casella, G. (2008). \emph{The Bayesian Lasso.}\cr
  Journal of the American Statistical Association, 103(482),
  June 2008, pp. 681-686\cr
  \url{http://www.stat.ufl.edu/~casella/Papers/Lasso.pdf}

  Griffin, J.E. and Brown, P.J. (2009).
  \emph{Inference with Normal-Gamma prior distributions in
    regression problems.} Bayesian Analysis, 5, pp. 171-188.\cr
  \url{http://projecteuclid.org/euclid.ba/1340369797}
  
  Hans, C. (2009). \emph{Bayesian Lasso regression.}
  Biometrika 96, pp. 835-845.\cr
  \url{http://biomet.oxfordjournals.org/content/96/4/835.abstract}

  Carvalho, C.M., Polson, N.G., and Scott, J.G. (2010) \emph{The
 horseshoe estimator for sparse signals.} Biometrika 97(2):
 pp. 465-480.\cr
  \url{http://ftp.stat.duke.edu/WorkingPapers/08-31.pdf}

  Geweke, J. (1996). \emph{Variable selection and model comparison
    in regression.} In Bayesian Statistics 5.  Editors: J.M. Bernardo,
  J.O. Berger, A.P. Dawid and A.F.M. Smith, 609-620. Oxford Press.

  Paul T. Troughton and Simon J. Godsill (1997).
  \emph{A reversible jump sampler for autoregressive time series,
    employing full conditionals to achieve efficient model space moves.}
  Technical Report CUED/F-INFENG/TR.304, Cambridge University
  Engineering Department.
  
  Geweke, J. (1993) \emph{Bayesian treatment of the independent
    Student-t linear model.} Journal of Applied Econometrics, Vol. 8,
    S19-S40
  
  \url{http://bobby.gramacy.com/r_packages/monomvn}
}

\author{ Robert B. Gramacy \email{rbg@vt.edu} }

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
legend("topleft", c("blasso-map", "lasso", "lsr"),
       col=c(2,2,3), pch=c(21,20,18))

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


\dontrun{
## fit with Student-t errors
## (~400-times slower due to automatic thinning level)
regt.blas <- blasso(x, y, theta=0.1)

## plotting some information about nu, and quantiles
plot(regt.blas, "nu", burnin=200)
quantile(regt.blas$nu[-(1:200)], c(0.05, 0.95))

## Bayes Factor shows strong evidence for Student-t model
mean(exp(regt.blas$llik[-(1:200)] - regt.blas$llik.norm[-(1:200)]))
}

## clean up
detach(diabetes)
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ regression }
