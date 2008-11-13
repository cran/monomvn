\name{default.QP}
\alias{default.QP}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Generating a default Quadratic Program for bmonomvn }
\description{
  This function generates a default \dQuote{minimum variance}
  Quadratic Program in order to obtain samples of the solution
  under the posterior for parameters \eqn{\mu}{mu} and \eqn{\Sigma}{S}
  obtained via \code{bmonomvn}.  The list generated as output
  has entries similar to the inputs of \code{\link[quadprog]{solve.QP}}
  from the \pkg{quadprog} package
}
\usage{
default.QP(m, dmu = FALSE, mu.constr = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{m}{ the dimension of the solution space; usually
    \code{ncol(y)} or equivalently \code{length(mu)}, \code{ncol(S)}
    and \code{nrow(S)} in the usage of \code{\link{bmonomvn}} }
  \item{dmu}{ a logical indicating whether \code{dvec} should
    be replaced with samples of \eqn{\mu}{mu}; see details below }
  \item{mu.constr}{ a vector indicating linear constraints on the
    samples of \eqn{\mu}{mu} to be included in the default constraint
    set. See details below; the default of \code{NULL} indicates none }
}
\details{
  When \code{\link{bmonomvn}(y, QP=TRUE)} is called, this
  function is used to generate a default Quadratic Program that
  samples from the argument \eqn{w}{w} such that
  \deqn{\min_w w^\top \Sigma w,}{min_w t(w) * S * w,}
    subject to the constraints that all
    \eqn{0\leq w_i \leq 1}{0 <= w[i] <=1}, for
    \eqn{i=1,\dots,m}{i=1:m},
    \deqn{\sum_{i=1}^m w_i = 1,}{sum(w) = 1,}
    and where \eqn{\Sigma}{S} is sampled from its posterior distribution
    conditional on the data \eqn{y}{y}.
  Alternatively, this function
  can be used as a skeleton to for adaptation to more general 
  Quadratic Programs by adjusting the \code{list} that is returned,
  as described in the \dQuote{value} section below.
  In the \dQuote{examples} section we set up a mean-variance and
  fixed-mean minimum variance Quadratic Program instead, via
  the \code{dmu} and \code{mu.constr} arguments.

  Non-default settings of the arguments \code{dmu} and \code{mu.constr}
  augment the default Quadratic Program, described above, in two standard ways.
  Specifying \code{dvec = TRUE} causes the program to change
  to \deqn{\min_w - w^\top \mu + \frac{1}{2} w^\top \Sigma w,}{min_w
    - t(w) * mu + 0.5 * t(w) * S * w,}
  with the same constraints as above.  Setting \code{mu.constr = 1},
  say, would augment the constraints to include
  \deqn{\mu^\top w \geq 1,}{t(mu) * w >= 1,} for samples of \eqn{\mu}{mu}
  from the posterior.  Setting \code{mu.constr = c(1,2)} would
  augment the constraints still further with
  \deqn{-\mu^\top w \geq -2,}{-t(mu) * w >= -2,} i.e., with
  alternating sign on the linear part, so that each sample of
  \eqn{\mu^\top w}{t(mu)*w} must lie in the interval [1,2].
  So whereas \code{dmu = TRUE} allows the \code{mu} samples to
  enter the objective in s standard way, \code{mu.constr}
  (\code{!= NULL}) allows it to enter the constraints
}
\value{
  This function returns a \code{list} that can be interpreted as
  specifying the following arguments to the
  \code{\link[quadprog]{solve.QP}} function in the \pkg{quadprog}
  package.  See \code{\link[quadprog]{solve.QP}} for more information
  of the general specification of these arguments.  In what follows
  we simply document the defaults provided by \code{default.QP}.
  Note that the \code{Dmat} argument is not specified as
  \code{\link{bmonomvn}} will use samples from \code{S} (from the
  posterior) instead
  
  \item{dvec }{ a zero-vector \code{rep(0, m)}, or a one-vector
    \code{rep(1, m)} when \code{dmu = TRUE} as the real \code{dvec}
    that will be used by \code{\link[quadprog]{solve.QP}} will
    then be \code{dvec * mu} }
  \item{dmu }{ a copy of the \code{dmu} input argument }
  \item{Amat }{ a \code{matrix} describing a linear transformation
    which, together with \code{b0}  and \code{meq}, describe the
    constraints that the components of the sampled solution(s),
    \code{w}, must be positive and sum to one }
  \item{b0 }{ a vector describing the in/equalities described by
    the these constraints }
  \item{meq}{ an integer scalar indicating that the first \code{meq}
    constraints described by \code{Amat} and \code{b0} are equality
    constraints; the rest are \code{>=} }
  \item{mu.constr}{ a vector whose length is one greater than the
    input argument of the same name, providing \code{bmonomvn} with
    the number 
    
    \code{mu.constr[1] = length(mu.constr[-1])}

    and location \code{mu.constr[-1]} of the columns of \code{Amat}
    which require multiplication by samples of \code{mu} }
}
%\references{ ~put references to the literature/web site here ~ }

\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk} }
\seealso{ \code{\link{bmonomvn}} and \code{\link[quadprog]{solve.QP}}
  in the \pkg{quadprog} package }
\examples{
## generate N=100 samples from a 10-d random MVN
## and randomly impose monotone missingness
xmuS <- randmvn(100, 20)
xmiss <- rmono(xmuS$x)

## set up the minimum-variance (default) Quadratic Program
## and sample from the posterior of the solution space
qp1 <- default.QP(ncol(xmiss))
obl1 <- bmonomvn(xmiss, QP=qp1)

## now obtain samples from the solution space of the
## mean-variance QP
qp2 <- default.QP(ncol(xmiss), dmu=TRUE)
obl2 <- bmonomvn(xmiss, QP=qp2)

## now obtain samples from minimum variance solutions
## where the mean weighted (samples) are constrained to be
## greater one
qp3 <- default.QP(ncol(xmiss), mu.constr=1)
obl3 <- bmonomvn(xmiss, QP=qp3)

## plot a comparison
par(mfrow=c(3,1))
plot(obl1, which="QP", xaxis="index", main="Minimum Variance")
plot(obl2, which="QP", xaxis="index", main="Mean Variance")
plot(obl3, which="QP", xaxis="index", main="Minimum Variance, mean >= 1")

## for a further comparison of samples of the QP solution
## w under Bayesian and non-Bayesian monomvn, see the
## examples in the bmonomvn help file
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.

\keyword{ optimize }
