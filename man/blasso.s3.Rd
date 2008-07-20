\name{blasso.s3}
\alias{print.blasso}
\alias{plot.blasso}
\alias{summary.blasso}
\alias{print.summary.blasso}
\title{ Summarizing Bayesian Lasso Output }

\description{
  Summarizing, printing, and plotting the contents of a
  \code{"blasso"}-class object containing samples from
  the posterior distribution of a Bayesian lasso model
}

\usage{
\method{print}{blasso}(x, ...)
\method{summary}{blasso}(object, burnin = 0, ...)
\method{plot}{blasso}(x, which=c("coef", "s2", "lambda2", "tau2i", "m"),
     burnin = 0, ... )
\method{print}{summary.blasso}(x, ...)
}

\arguments{
  \item{object}{ a \code{"blasso"}-class object for that must be named
    \code{object} for the generic methods \code{\link{summary.blasso}} }
  \item{x}{ a \code{"blasso"}-class object that must be named \code{x}
    for the generic printing and plotting methods 
    \code{\link{print.summary.blasso}} and
    \code{\link{plot.blasso}} }
  \item{burnin}{ number of burn-in rounds to discard before
    reporting summaries and making plots.  Must be non-negative
    and less than \code{x$T}}
  \item{which}{ indicates the parameter whose characteristics should be plotted;
     does not apply to the \code{\link{summary}} }
  \item{...}{ passed to \code{\link{print.blasso}}, or
      \code{\link{plot.default}}}
  }
  
\details{
  \code{\link{print.blasso}} prints the \code{call} followed by a
  brief summary of the MCMC run and a suggestion to try
  the summary and plot commands.  

  \code{\link{plot.blasso}} uses an appropriate
  \code{\link{plot}} command
  on the list entries of the \code{"blasso"}-class object thus
  visually summarizing the samples from the posterior distribution of
  each parameter in the model depending on the \code{which}
  argument supplied.
  
  \code{\link{summary.blasso}} uses the \code{\link{summary}} command
  on the list entries of the \code{"blasso"}-class object thus
  summarizing the samples from the posterior distribution of each
  parameter in the model.

  \code{\link{print.summary.monomvn}} calls \code{\link{print.blasso}}
  on the \code{object} and then prints the result of
  \code{\link{summary.blasso}}
}
\value{
  \code{\link{summary.blasso}} returns a \code{"summary.blasso"}-class
  object, which is a list containing (a subset of) the items below.
  The other functions do not return values.

  \item{B}{ a copy of the input argument \code{thin} }
  \item{T}{ total number of MCMC samples to be collected from \code{x$T}
  }
  \item{thin}{ number of MCMC samples to skip before a sample is
    collected (via thinning) from \code{x$T} }
  \item{coef}{ a joint \code{summary} of \code{x$mu} and
    the columns of \code{x$beta}}
  \item{s2}{ a \code{summary} of \code{x$s2}}
  \item{lambda2}{ a \code{summary} of \code{x$lambda2} when lasso
    or ridge regression is active }
  \item{tau2}{ a \code{summary} of the columns of \code{x$tau2i} when
    lasso is active }
  \item{bn0}{ the estimated posterior probability that the individual
	components of the regression coefficients \code{beta} is 
 	nonzero  }
}

\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk} }

\references{
\url{http://www.statslab.cam.ac.uk/~bobby/monomvn.html}
}

\seealso{ \code{\link{blasso}} }
\keyword{ methods }
\keyword{ hplot }
