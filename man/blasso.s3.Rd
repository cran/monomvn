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
\method{plot}{blasso}(x, which=c("coef", "s2", "lambda2", "gamma",
    "tau2i","omega2", "nu", "m", "pi"), subset = NULL, burnin = 0,
    ... )
\method{print}{summary.blasso}(x, ...)
}

\arguments{
  \item{object}{ a \code{"blasso"}-class object that must be named
    \code{object} for the generic methods \code{\link{summary.blasso}} }
  \item{x}{ a \code{"blasso"}-class object that must be named \code{x}
    for the generic printing and plotting methods 
    \code{\link{print.summary.blasso}} and
    \code{\link{plot.blasso}} }
  \item{subset}{ a vector of indicies that can be used to specify
    the a subset of the columns of \code{tau2i} or \code{omega2} that
    are plotted as boxplots in order to reduce clutter }
  \item{burnin}{ number of burn-in rounds to discard before
    reporting summaries and making plots.  Must be non-negative
    and less than \code{x$T}}
  \item{which}{ indicates the parameter whose characteristics
    should be plotted; does not apply to the \code{\link{summary}} }
  \item{...}{ passed to \code{\link{print.blasso}}, or
      \code{\link{plot.default}}}
  }
  
\details{
  \code{\link{print.blasso}} prints the \code{call} followed by a
  brief summary of the MCMC run and a suggestion to try
  the summary and plot commands.  

  \code{\link{plot.blasso}} uses an appropriate
  \code{\link{plot}} command on the \code{list} entries of the
  \code{"blasso"}-class object thus
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
  object, which is a \code{list} containing (a subset of) the items below.
  The other functions do not return values.

  \item{B}{ a copy of the input argument \code{thin} }
  \item{T}{ total number of MCMC samples to be collected from \code{x$T}
  }
  \item{thin}{ number of MCMC samples to skip before a sample is
    collected (via thinning) from \code{x$T} }
  \item{coef}{ a joint \code{summary} of \code{x$mu} and
    the columns of \code{x$beta}, the regression coefficients }
  \item{s2}{ a \code{summary} of \code{x$s2}, the variance parameter }
  \item{lambda2}{ a \code{summary} of \code{x$lambda2}, the penalty
    parameter, when lasso or ridge regression is active }
  \item{lambda2}{ a \code{summary} of \code{x$gamma},
    when the NG extensions to the lasso are used }
  \item{tau2i}{ a \code{summary} of the columns of the latent
    \code{x$tau2i} parameters when lasso is active }
  \item{omega2}{ a \code{summary} of the columns of the latent
    \code{x$omega2} parameters when Student-t errors are active }
  \item{nu}{ a \code{summary} of \code{x$nu}, the degrees of freedom
    parameter, when the Student-t model is active }
  \item{bn0}{ the estimated posterior probability that the individual
	components of the regression coefficients \code{beta} is 
 	nonzero  }
  \item{m}{ a \code{summary} the model order \code{x$m}: the
	number of non-zero regression coefficients \code{beta} }
  \item{pi}{ the estimated Binomial proportion in the prior for
    the model order when 2-vector input is provided for
    \code{mprior}
  }
}

\author{ Robert B. Gramacy \email{rbgramacy@chicagobooth.edu} }

\references{
\url{http://faculty.chicagobooth.edu/robert.gramacy/monomvn.html}
}

\seealso{ \code{\link{blasso}} }
\keyword{ methods }
\keyword{ hplot }
