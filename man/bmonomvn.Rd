\name{bmonomvn}
\alias{bmonomvn}

\title{Bayesian Estimation for Multivariate Normal Data with
  Monotone Missingness}

\description{
  Bayesian estimation via sampling from the posterior distribution of the
  of the mean and covariance matrix of multivariate normal (MVN)
  distributed data with a monotone missingness pattern, via Gibbs Sampling.
  Through the use of parsimonious/shrinkage regressions (lasso & ridge),
  where standard regressions fail, this function can handle an (almost)
  arbitrary amount of missing data
}

\usage{
bmonomvn(y, pre = TRUE, p = 0.9, B = 100, T = 200, thin = 1,
         economy = FALSE, method = c("lasso", "ridge", "lsr", "factor"),
         RJ = c("bpsn", "p", "none"), capm = method!="lasso",
         start = NULL, mprior = 0, rd = NULL, rao.s2 = TRUE,
         QP = NULL, verb = 1,trace = FALSE)
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
   alternative regression (lasso, ridge, or RJ) is performed as if
   least-squares regression has \dQuote{failed}.  Least-squares regression is
   known to fail when the number of columns equals the number of rows,
   hence a default of \code{p = 0.9 <= 1}. Alternatively, setting
   \code{p = 0} forces a parsimonious method to be used for
   \emph{every} regression. Intermediate settings of \code{p} allow
   the user to control when least-squares regressions stop and the
   parsimonious ones start;  When \code{method = "factor"} the \code{p}
   argument represents an integer (positive) number of initial columns
   of \code{y} to treat as known factors}
 \item{B}{ number of Burn-In MCMC sampling rounds,
   during which samples are discarded }
  \item{T}{ total number of MCMC sampling rounds to take
    place after burn-in, during which samples are saved }
  \item{thin}{ multiplicative thinning in the MCMC.  Each Bayesian
    (lasso) regression will discard \code{thin*M} MCMC rounds,
    where \code{M} is the number of columns in its design matrix,
    before a sample is saved as a draw from the posterior distribution }
  \item{economy }{ indicates whether memory should be economized at
    the expense of speed.  When \code{TRUE} the individual Bayesian
    (lasso) regressions are cleaned between uses so that only one
    of them has a large footprint at any time during sampling from
    the Markov chain.  When \code{FALSE} (default) all regressions
    are pre-allocated and the full memory footprint is realized at
    the outset, saving dynamic allocations}
  \item{method}{ indicates the Bayesian parsimonious regression
    specification to be used, choosing between the lasso (default)
    of Park \& Casella, a ridge regression special case of the lasso,
    and least-squares. The \code{"factor"} method treats the first
    \code{p} columns of \code{y} as known factors}
  \item{RJ }{ indicates the Reversible Jump strategy to be employed.
    The default argument of \code{"bpsn"} only uses RJ for
    regressions with \code{p >= n}.  The \code{"p"} method uses
    RJ whenever a parsimonious regression is used, and \code{"none"}
    never uses RJ}
  \item{capm}{ when \code{TRUE} this argument indicates that the
    number of components of
    \eqn{\beta}{beta} should not exceed \eqn{n}{n}, the number of
    response variables in a particular regression.  This argument
    is ignored when using \code{method = "lasso"} }
  \item{start}{ a list depicting starting values for the parameters
    that are use to initialize the Markov chain.  Usually this will be
    a \code{"monomvn"}-class object depicting maximum likelihood
    estimates output from the \code{\link{monomvn}} function.
    The relevant fields are the mean vector \code{$mu}, covariance
    matrix \code{$S}, monotone ordering \code{$o} (for sanity checking
    with input \code{y}), component vector \code{$ncomp} and
    penalty parameter vector \code{$lambda}; see note below }
  \item{mprior}{ prior on the number of non-zero regression coefficients
    (and therefore covariates) \code{m} in the model. The default
    (\code{mprior = 0}) encodes the uniform prior on \code{0 < m < M}.
    A scalar value \code{0 <= mprior <= 1} implies a Binomial prior
    \code{Bin(m|n=M,p=mprior)}.  A 2-vector \code{mprior=c(g,h)}
    of positive values \code{g} and \code{h} represents
    gives \code{Bin(m|n=M,p)} prior where \code{p~Beta(g,h)}}
  \item{rd}{ \code{=c(r,delta)}; a 2-vector of prior parameters for
    \eqn{\lambda^2}{lambda^2}
    which depends on the regression method.  When \code{method =
      "lasso"} then the components are the \eqn{\alpha}{alpha}
    (shape) and \eqn{\beta}{beta} (rate) parameters to the a
    gamma distribution \code{G(r,delta)};
    when \code{method = "ridge"} the components are the
    \eqn{\alpha}{alpha} (shape) and \eqn{\beta}{beta} (scale) parameters
    to an inverse-gamma distribution \code{IG(r/2,delta/2)} }
  \item{rao.s2 }{indicates whether to  Rao-Blackwellized samples for
    \eqn{\sigma^2}{s^2} should be used (default \code{TRUE}); see
    the details section of \code{\link{blasso}} for more information }
  \item{QP }{ if non-\code{NULL} this argument should either be
    \code{TRUE}, a positive integer, or contain a \code{list}
    specifying a Quadratic Program to solve as a function of the
    samples of \code{mu = dvec} and
    \code{Sigma = Dmat} in the notation of \code{\link[quadprog]{solve.QP}};
    see \code{\link{default.QP}} for a default specification that
    is used when \code{QP = TRUE} or a positive integer is is given;
    more details are below }
  \item{verb}{ verbosity level; currently only \code{verb = 0} and
    \code{verb = 1} are supported }
  \item{trace}{ if \code{TRUE} then samples from all parameters are
    saved to files in the CWD, and then read back into the
    \code{"monomvn"}-class object upon return }
}

\details{
  If \code{pre = TRUE} then \code{bmonomvn} first re-arranges the columns
  of \code{y} into nondecreasing order with respect to the number of
  missing (\code{NA})  entries. Then (at least) the first column should
  be completely observed.

  Samples from the posterior distribution of the MVN mean vector and
  covariance matrix are obtained sampling
  from the posterior distribution of Bayesian regression models.
  The methodology for converting these to samples from the mean vector
  and covariance matrix is outlined in the \code{\link{monomvn}}
  documentation, detailing a similarly structured maximum likelihood
  approach.  Also see the references below.

  Whenever the regression model is ill--posed (i.e., when there are
  more covariates than  responses, or a
  \dQuote{big \code{p} small \code{n}} problem) then
  Bayesian lasso or ridge regressions -- possibly augmented with Reversible
  Jump (RJ) for model selection -- are used instead.
  See the Park \& Casella reference below, and the \code{\link{blasso}}
  documentation.  To guarantee each regression is well posed the
  combination setting of \code{method="lsr"} and \code{RJ="none"}
  is not allowed.
  As in \code{\link{monomvn}} the \code{p} argument can be used to
  turn on lasso or ridge regressions (possibly with RJ) at other times.
  The exception is the \code{"factor"} method which always involves
  an OLS regression on (a subset of) the first \code{p}
  columns of \code{y}.

  Samples from a function of samples of \code{mu} and \code{Sigma}
  can be obtained by specifying a Quadratic program via the
  argument \code{QP}.  The idea is to allow for the calculation of
  the distribution of minimum variance and mean--variance portfolios,
  although the interface is quite general.  See \code{\link{default.QP}}
  for more details, as \code{\link{default.QP}(ncol(y))} is used
  when the argument \code{QP = TRUE} is given.  When a positive integer
  is given, then the first \code{QP} columns of \code{y} are treated
  as factors by using

  \code{\link{default.QP}(ncol(y) - QP)}

  instead. The result is that the corresponding components of (samples of)
  \code{mu} and rows/cols of \code{S} are not factored into the
  specification of the resulting Quadratic Program
}

\value{
  \code{bmonomvn} returns an object of class \code{"monomvn"},
  which is a \code{list} containing the inputs above and a
  subset of the components below.

  \item{call }{a copy of the function call as used}
  \item{mu }{estimated mean vector with columns corresponding to the
    columns of \code{y} }
  \item{S }{estimated covariance matrix with rows and columns
    corresponding to the columns of \code{y} }
  \item{mu.var }{estimated variance of the mean vector with columns
    corresponding to the columns of \code{y} }
  \item{S.var }{estimated variance of the individual components of the
    covariance matrix with columns and rows corresponding to the columns
    of \code{y}}
  \item{mu.map }{estimated maximum \emph{a' posteriori} (MAP) of the
    mean vector with columns corresponding to the columns of \code{y} }
  \item{S.map }{estimated MAP of the individual
    components of the covariance matrix with columns and rows
    corresponding to the columns of \code{y}}
  \item{lpost.map}{ log posterior probability of the MAP estimate}
  \item{which.map}{ gives the time index of the sample corresponding to
    the MAP estimate}
  \item{na }{ when \code{pre = TRUE} this is a vector containing number of
    \code{NA} entries in each column of \code{y}}
  \item{o }{ when \code{pre = TRUE} this is a vector containing the
    index of each column in the sorting of the columns of \code{y}
    obtained by \code{o <- \link{order}(na)}}
  \item{method }{method of regression used on each column, or
    \code{"bcomplete"} indicating that no regression was used}
  \item{thin }{ the number of thinning rounds used for the regression
    (\code{method}) column}
  \item{lambda2 }{ records the mean \eqn{\lambda^2}{lambda^2} value
    found in the trace of the Bayesian Lasso regressions.  Zero-values
    result when the column corresponds to a complete
    case or an ordinary least squares regression (these would be
    \code{NA} entries from \code{\link{monomvn}}) }
  \item{ncomp }{ records the mean number of components
    (columns of the design matrix) used in the regression model for
    each column of \code{y}. If input \code{RJ = FALSE} then this simply
    corresponds to the monotone ordering (these would correspond to
    the \code{NA} entries from \code{\link{monomvn}}).  When \code{RJ = TRUE}
    the monotone ordering is an upper bound (on each entry)}
  \item{trace}{ if input \code{trace = TRUE} then this field contains
    traces of the samples of \eqn{\mu}{mu} in the field \code{$mu} and
    of \eqn{\Sigma}{S} in the field \code{$S}, and of all regression
    parameters for each of the \code{m = length(mu)} columns in the field
    \code{$reg}. This \code{$reg} field is a stripped-down
    \code{"blasso"}-class object so that the methods of that object
    may be used for analysis.  If data augmentation is required to
    complete the monotone missingness pattern, then samples from these
    entries of \code{Y} are contained in \code{$DA} where the column
    names indicate the \code{i}-\code{j} entry of \code{Y} sampled;
    see the \code{R} output below }
  \item{R}{ gives a \code{matrix} version of the missingness pattern
    used: \code{0}-entries mean observed; \code{1}-entries indicate
    missing values conforming to a monotone pattern; \code{2}-entries
    indicate missing values that require data augmentation to complete
    a monotone missingness pattern }
  \item{B}{ from inputs: number of Burn-In MCMC sampling rounds,
    during which samples are discarded }
  \item{T}{ from inputs: total number of MCMC sampling rounds to take
    place after burn-in, during which samples are saved }
  \item{r}{ from inputs: alpha (shape) parameter to the gamma
      distribution prior for the lasso parameter lambda }
  \item{delta}{ from inputs: beta (rate) parameter to the gamma
    distribution prior for the lasso parameter lambda }
  \item{QP}{ if a valid (non--\code{FALSE} or \code{NULL}) \code{QP}
    argument is given, then this field contains the specification of
    a Quadratic Program in the form of a list with entries including
    \code{$dvec}, \code{$Amat}, \code{$b0}, and \code{$meq}, similar
    to the usage in \code{\link[quadprog]{solve.QP}}, and some
    others; see \code{\link{default.QP}} for more details}
  \item{W}{ when input \code{QP = TRUE} is given, then this field
    contains a \code{T*ncol(y)} matrix of samples from the posterior
    distribution of the solution to the Quadratic Program, which can
    be visualized via \code{\link{plot.monomvn}} using the argument
    \code{which = "QP"} }
}

\references{
  Robert B. Gramacy, Joo Hee Lee, and Ricardo Silva (2008).
  \emph{On estimating covariances between many assets with histories 
  of highly variable length}. \cr Preprint available on arXiv:0710.5837:
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
  where \code{p >= n}, i.e., if any of the columns in the \code{y}
  matrix have fewer non--\code{NA} elements than the number of
  columns with more non--\code{NA} elements, then it is helpful
  to employ both lasso/ridge and the RJ method.

  It is important that any starting values provided in the
  \code{start} be compatible with the regression model
  specified by inputs \code{RJ} and \code{method}.  Any
  incompatibilities will result with a warning that
  (alternative) default action was taken and may result in
  an undesired (possibly inferior) model being fit
}

\seealso{ \code{\link{blasso}}, \code{\link{monomvn}},
  \code{\link{default.QP}}, \code{\link[norm]{em.norm}} in the
  \pkg{norm} package, and \code{\link[mvnmle]{mlest}} in the
  \pkg{mvnmle} package, \code{\link{returns}}}

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

## generate N=100 samples from a 10-d random MVN
xmuS <- randmvn(100, 20)

## randomly impose monotone missingness
xmiss <- rmono(xmuS$x)

## using least squares only when necessary,
obl <- bmonomvn(xmiss)
obl

## look at the posterior variability
par(mfrow=c(1,2))
plot(obl)
plot(obl, "S")

## compare to maximum likelihood
Ellik.norm(obl$mu, obl$S, xmuS$mu, xmuS$S)
oml <- monomvn(xmiss, method="lasso")
Ellik.norm(oml$mu, oml$S, xmuS$mu, xmuS$S)

##
## a min-variance portfolio allocation example
##

## get the returns data, and use 20 random cols
data(returns)
train <- returns[,sample(1:ncol(returns), 20)]

## missingness pattern requires DA; also gather
## samples from the solution to a QP
obl.da <- bmonomvn(train, p=0, QP=TRUE)

## plot the QP weights distribution
plot(obl.da, "QP", xaxis="index")

## get ML solution: will warn about monotone violations
suppressWarnings(oml.da <- monomvn(train, method="lasso"))

## add mean and MLE comparison, requires the
## quadprog library for the solve.QP function
add.pe.QP(obl.da, oml.da)

## now consider adding in the market as a factor
data(market)
mtrain <- cbind(market, train)

## fit the model using only factor regressions
obl.daf <- bmonomvn(mtrain, method="factor", p=1, QP=1)
plot(obl.daf, "QP", xaxis="index", main="using only factors")
suppressWarnings(oml.daf <- monomvn(mtrain, method="factor"))
add.pe.QP(obl.daf, oml.daf)


##
## a Bayes/MLE comparison using least squares sparingly
##

## fit Bayesian and classical lasso
obls <- bmonomvn(xmiss, p=0.25)
Ellik.norm(obls$mu, obls$S, xmuS$mu, xmuS$S)
omls <- monomvn(xmiss, p=0.25, method="lasso")
Ellik.norm(omls$mu, omls$S, xmuS$mu, xmuS$S)

## compare to ridge regression
obrs <- bmonomvn(xmiss, p=0.25, method="ridge")
Ellik.norm(obrs$mu, obrs$S, xmuS$mu, xmuS$S)
omrs <- monomvn(xmiss, p=0.25, method="ridge")
Ellik.norm(omrs$mu, omrs$S, xmuS$mu, xmuS$S)

## using the maximum likelihood solution to initialize
## the Markov chain and avoid burn-in.  
ob2s <- bmonomvn(xmiss, p=0.25, B=0, start=omls, RJ="p")
Ellik.norm(ob2s$mu, ob2s$S, xmuS$mu, xmuS$S)
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ multivariate }
\keyword{ regression }
\keyword{ optimize }
