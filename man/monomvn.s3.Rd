\name{monomvn.s3}
\alias{print.monomvn}
\alias{summary.monomvn}
\alias{print.summary.monomvn}
\alias{plot.summary.monomvn}
\title{ Summarizing monomvn output }
\description{
  Summarizing, printing, and plotting the contents of a
  \code{"monomvn"}-class object
}
\usage{
\method{summary}{monomvn}(object, Si = FALSE, ...)
\method{print}{summary.monomvn}(x, ...)
\method{plot}{summary.monomvn}(x, gt0 = FALSE, main = NULL,
     xlab = "number of zeros", ...)
}
\arguments{
  \item{object}{ a \code{"monomvn"}-class object that must be named
    \code{object} for the generic methods \code{\link{summary.monomvn}} }
  \item{x}{ a \code{"monomvn"}-class object that must be named \code{x}
    for generic printing and plotting via
    \code{\link{print.summary.monomvn}} and
    \code{\link{plot.summary.monomvn}} }
  \item{Si}{ boolean indicating whether \code{object$S} should be
    inverted and inspected for zeros within
\code{\link{summary.monomvn}}, indicating pairwise independence;
    default is \code{FALSE} }
  \item{gt0}{ boolean indicating whether the histograms in
    \code{\link{plot.summary.monomvn}} should exclude columns
    of \code{object$S} or \code{Si} without any zero entries }
  \item{main}{ optional text to be added to the main title of the histograms
    produced by the generic \code{\link{plot.summary.monomvn}} }
  \item{xlab}{label for the x-axes of the histograms produced by
    \code{\link{plot.summary.monomvn}}; otherwise default
    automatically-generated text is used }
  \item{...}{ passed to \code{\link{print.monomvn}}, or
      \code{\link{plot.default}}}
}
\details{
  These functions work on the output from both \code{\link{monomvn}}
  and \code{\link{bmonomvn}}.

  \code{\link{print.monomvn}} prints the \code{call} followed by a
  summary of the regression  method used at each iteration of the
  algorithm.  It also indicates how many completely observed features
  (columns) there were in the data.
  For non-least-squares regressions (i.e., \pkg{plsr}, \pkg{lars}
  and \code{\link[MASS]{lm.ridge}} methods)
  and indication of the method used for selecting the
  number of components (i.e., \code{CV}, \code{LOO}, etc., or
  \code{none}) is provided

  \code{\link{summary.monomvn}} summarizes information about the
  number of zeros in the estimated covariance matrix \code{object$S}
  and its inverse

  \code{\link{print.summary.monomvn}} calls \code{\link{print.monomvn}}
  on the \code{object} and then prints the result of
  \code{\link{summary.monomvn}}

  \code{\link{plot.summary.monomvn}} makes histograms of the number of
  zeros in the columns of \code{object$S} and its inverse
}
\value{
  \code{\link{summary.monomvn}} returns a
  \code{"summary.monomvn"}-class object, which is a \code{list}
  containing (a subset of) the items below.  The other
  functions do not return values.

  \item{obj}{ the \code{"monomvn"}-class \code{object} }
  \item{marg}{ the proportion of zeros in \code{object$S} }
  \item{S0}{ a vector containing the number of zeros in each column
    of \code{object$S} }
  \item{cond}{ if input \code{Si = TRUE} this field contains the
    proportion of zeros in the inverse of \code{object$S} }
  \item{Si0}{ if input \code{Si = TRUE} this field contains a
      vector with the number of zeros in each column of the inverse
      of \code{object$S} }
}
\author{ Robert B. Gramacy \email{rbg@vt.edu} }
\references{
\url{https://bobby.gramacy.com/r_packages/monomvn/}
}
\note{
  There is one further S3 function for \code{"monomvn"}-class
  objects that has its own help file: \code{\link{plot.monomvn}}
}
\seealso{ \code{\link{bmonomvn}}, \code{\link{monomvn}},
  \code{\link{plot.monomvn}} }
\keyword{ methods }
\keyword{ hplot }
