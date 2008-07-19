\name{bmonomvn}
\alias{bmonomvn}

\title{Bayesian Estimation for Multivariate Normal Data with
  Monotone Missingness}

\description{
  Bayesian estimation via sampling from the posterior distribution of the
  of the mean and covariance matrix of multivariate normal (MVN)
  distributed data with a monotone missingness pattern via Gibbs Sampling.
  Through the use of parsimonious/shrinkage regressions (currently only
  lasso is supported), where standard regressions fail,
  this function can handle an (almost) arbitrary amount of missing data.
}

\usage{
bmonomvn(y, pre = TRUE, p = 0.9, B = 100, T = 200, thin = 10,
         method = c("default", "rjlasso", "rjlsr", "lasso"),
         capm = method!="lasso", start = NULL, r = 2, delta = 0.1,
         rao.s2 = TRUE, verb = 1, trace = FALSE)
}

\arguments{
 \item{y}{ data \code{matrix}  were each row is interpreted as a
   random sample from a MVN distribution with missing
   values indicated by \code{NA}}
 \item{pre}{ logical indicating whether pre-processing of the
   \code{y} is to be performed.  This sorts the columns so that the
   number of \code{NA}s is non-decreasing with the column index }
 \item{p}{ when performing regressions, \code{p} is the proportion of the
   number of columns to rows in the design matrix before an
   alternative regression (lasso) is performed as if
   least-squares regression \dQuote{failed}.  Least-squares regression is
   known to fail when the number of columns equals the number of rows,
   hence a default of \code{p = 0.9} close to \code{1}.
   Alternatively, setting \code{p = 0}
   forces lasso to be used for \emph{every} regression.
   Intermediate settings of \code{p} allow the user to control when
   least-squares regressions stop and the lasso ones start}
 \item{B}{ number of Burn-In MCMC sampling rounds,
   during which samples are discarded }
  \item{T}{ total number of MCMC sampling rounds to take
    place after burn-in, during which samples are saved }
  \item{thin}{ level of thinning in the MCMC, i.e., the number of MCMC
    rounds must be collected before a sample is saved }
  \item{method}{ indicates the lasso regression specification
    to be used.  The \code{"default"} uses the Trevor & Park formulation
    until \eqn{p \geq n}{p >= n} at which point a Reversible Jump (RJ)
    is turned on in order to keep the number of non-zero regression
    coefficients \eqn{\beta}{beta} at most \eqn{n}{n} (when \code{capm =
      TRUE}.  The \code{"rjlasso"} method uses RJ at all times that
    the lasso is used, whereas \code{"rjlsr"} uses RJ without the
    lasso prior.  Finally, the \code{"lasso"} method never uses RJ.
    See \code{\link{blasso}} and note below for more details}
  \item{capm}{ when \code{TRUE} this argument indicates that the
    number of components of
    \eqn{\beta}{beta} should not exceed \eqn{n}{n}, the number of
    response variables in a particular regression.  This argument
    is ignored when using \code{method = "lasso"} }
  \item{start}{ a list depicting starting values for the parameters
    that are use to initaliza the Markov chain.  Usually this will be
    a \code{"monomvn"}-class object depsicting maximum likelihood
    estimates output from the \code{\link{monomvn}} function, but
    the relevant fields are the mean vector \code{$mu}, covariance
    matrix \code{$S}, monotone ordering \code{$o} (for sanity checking
    with intput \code{y}), component vector \code{$ncomp} and
    penalty parameter vector \code{$lambda}.  See note below }
  \item{r}{ alpha (shape) parameter to the gamma distribution prior for
  the lasso parameter lambda }
  \item{delta}{ beta (rate) parameter to the gamma distribution prior
    for the lasso parameter lambda }
  \item{rao.s2 }{indicates whether to use Rao-Blackwellized samples for
    \eqn{\sigma^2}{s^2} should be used (default \code{TRUE}), see
    the details section of \code{\link{blasso}} for more information }
  \item{verb}{ verbosity level; currently only \code{verb = 0} and
    \code{verb = 1} are supported }
  \item{trace}{ if \code{TRUE} then samples from the regressions are
    saved to files in the CWD }
}

\details{
   If \code{pre = TRUE} then \code{bmonomvn} first re-arranges the columns
  of \code{y} into nondecreasing order with respect to the number of
  missing (\code{NA})  entries. Then (at least) the first column should
  be completely observed.

  Samples from the posterior distribution of the over a Multivariate
  Normal mean vector and covariance matrix are obtained by obtaining
  samples from the posterior distribution of Bayesian regression models.
  The method by which these samples from the regression posterior(s) are
  used to obtain samples from the mean vector and covariance matrix
  is outlined in the \code{\link{monomvn}} documentation, detailing a
  similarly structured maximum likelihood approach.  See also the
  references below.

  Whenever the regression model is ill--posed (i.e., when there are
  more  covariates than  responses, or a
  \dQuote{big \code{p} small \code{n}} problem) then
  Bayesian Lasso regressions -- possibly augmented with Reversible
  Jump (RJ) for model selection -- are used instead.
  See the Park \& Casella reference below, and the \code{\link{blasso}}
  documentation.  The \code{p} argument
  can be used to turn on Lasso regressions at other times.

  One difference between the Bayesian and MLE approach is that the
  MLE approach treats the complete (fully observed) columns
  jointly, without performing regressions.  In contrast, the Bayesian
  only treats the first complete column without regressions.  The
  remaining complete columns are processed via regression.
}

\value{
  \code{bmonomvn} returns an object of class \code{"monomvn"}, which is a
  list containing a subset of the components below.

  \item{call }{a copy of the function call as used}
  \item{mu }{estimated mean vector with columns corresponding to the
    columns of \code{y} }
  \item{S }{estimated covariance matrix with rows and columns
    corresponding to the columns of \code{y} }
  \item{mu.var }{estimated variance of mean vector with columns
    corresponding to the columns of \code{y} }
  \item{S.var }{estimated variance of the individual components of the
    covariance matrix with columns corresponding to the columns
    of \code{y}}
  \item{na }{ when \code{pre = TRUE} this is a vector containing number of
    \code{NA} entries in each column of \code{y}}
  \item{o }{ when \code{pre = TRUE} this is a vector containing the
    index of each column in the sorting of the columns of \code{y}
    obtained by \code{o <- \link{order}(na)}}
  \item{method }{method of regression used on each column, or
    \code{"bcomplete"} indicating that no regression was used}
  \item{lambda2 }{ records the mean \eqn{\lambda^2}{lambda^2} value
    found in the trace of the Bayesian Lasso regressions.  This value
    will be zero if the corresponding column corresponds to a complete
    case or a ordinary least squares regression (these would be
    \code{NA} entries maximum likelihood estimates are sought
    via \code{\link{monomvn}} }
  \item{ncomp }{ records the mean number of components
    (columns of the design matrix) used in the regression model for
    each column of \code{y}. If input \code{RJ = FALSE} then this simply
    corresponds to the monotone ordering (these would correspond to
    the \code{NA} entries when maximum likelihood estimates are sought
    via \code{\link{monomvn}}.  When \code{RJ = TRUE}
    the monotone ordering is an upper bound (on each entry)}
  \item{trace}{ if input \code{trace = TRUE} then this field contains
    traces of the samples of \code{mu} in the field \code{$mu} and
    of \code{S} in the field \code{$S}, and of all regression parameters
    for each of the \code{m = length(mu)} columns in the field
    \code{$reg} }
  \item{B}{ from inputs: number of Burn-In MCMC sampling rounds,
    during which samples are discarded }
  \item{T}{ from inputs: total number of MCMC sampling rounds to take
    place after burn-in, during which samples are saved }
  \item{r}{ from inputs: alpha (shape) parameter to the gamma
      distribution prior for the lasso parameter lambda }
  \item{delta}{ from inputs: beta (rate) parameter to the gamma
      distribution prior for the lasso parameter lambda }
}

\references{
  Robert B. Gramacy and Joo Hee Lee (2007).
  \emph{On estimating covariances between many assets with histories 
  of highly variable length}. Preprint available on arXiv:0710.5837:\cr
  \url{http://arxiv.org/abs/0710.5837}

  Roderick J.A. Little and Donald B. Rubin (2002).
  \emph{Statistical Analysis with Missing Data}, Second Edition.
  Wilely.

  Park, T., Casella, G. (2008).
  \emph{The Bayesian Lasso}, (unpublished)\cr
  \url{http://www.stat.ufl.edu/~casella/Papers/bayeslasso.pdf}

\url{http://www.statslab.cam.ac.uk/~bobby/monomvn.html}
}

\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk} }

\note{
  Whenever the \code{bmonomvn} algorithm requires a regression
  where \code{p >= n}, i.e., if any of the columns in the \code{Y}
  matrix have fewer non--\code{NA} elements than the number of
  columns with more non--\code{NA} elements. then it is helpful
  to employ both lasso and RJ method.  Therefore, should this
  case be detected and one of the \code{"rjlsr"} or \code{"lasso"}
  methods is specified, a warning will be printed

  It is important than any starting values provided in the
  \code{start} input list be compatible with the regression model
  specified by inputs \code{RJ} and \code{method}.  Any
  incompatibilites will result with a warning that
  (alternative) default action was taken and may result in
  an undesired (possibly inferior) model being fit
}

\seealso{ \code{\link{blasso}}, \code{\link{monomvn}},
  \code{\link[norm]{em.norm}} in the \pkg{norm} package,
  and \code{\link[mvnmle]{mlest}} in the \pkg{mvnmle} package}

\examples{
## standard usage, duplicating the results in
## Little and Rubin, section 7.4.3
data(cement.miss)
out <- bmonomvn(cement.miss)
out
out$mu
out$S

##
## A bigger example, comparing the various methods
##

## generate N=1000 samples from a random MVN
xmuS <- randmvn(1000, 100)

## randomly impose monotone missingness
xmiss <- rmono(xmuS$x)

## using least squares only when necessary
out.b <- bmonomvn(xmiss)
out.b
kl.norm(out.b$mu, out.b$S, xmuS$mu, xmuS$S, symm=TRUE)
out.mle <- monomvn(xmiss, method="lasso")
kl.norm(out.mle$mu, out.mle$S, xmuS$mu, xmuS$S, symm=TRUE)

## using least squares sparingly
out.b.s <- bmonomvn(xmiss, p=0.25)
kl.norm(out.b.s$mu, out.b.s$S, xmuS$mu, xmuS$S, symm=TRUE)
out.mle.s <- monomvn(xmiss, p=0.25, method="lasso")
kl.norm(out.mle.s$mu, out.mle.s$S, xmuS$mu, xmuS$S, symm=TRUE)

## using the maximum likelihood solution to initialize
## the Markov chain and avoid burn-in.  
out.b2.s <- bmonomvn(xmiss, p=0.25, B=0, start=out.mle.s, method="rjlasso")
kl.norm(out.b2.s$mu, out.b2.s$S, xmuS$mu, xmuS$S, symm=TRUE)
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ multivariate }
\keyword{ regression }
