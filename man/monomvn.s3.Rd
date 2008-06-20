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
  \item{object}{ a \code{"monomvn"}-class object for that must be named
    \code{object} for the generic methods \code{\link{summary.monomvn}} }
  \item{x}{ a \code{"monomvn"}-class object that must be named \code{x}
    for generic printing and plotting via
    \code{\link{print.summary.monomvn}} and
    \code{\link{plot.summary.monomvn}} }
  \item{Si}{ boolean indicating whether \code{object$S} should be
    inverted within top look for zeros within
\code{\link{summary.monomvn}}, indicating pairwise independence;
    default is \code{FALSE} }
  \item{gt0}{ boolean indicating whether the histograms in
    \code{\link{plot.summary.monomvn}} should exclude columns
    of \code{object$S} or \code{Si} without any zero entries }
  \item{main}{ text to be added to the main title of the histograms
    produced by the generic \code{\link{plot.summary.monomvn}} }
  \item{xlab}{label for the x-axes of the histograms produced by
    \code{\link{plot.summary.monomvn}} }
  \item{...}{ passed to \code{\link{print.monomvn}}, or
      \code{\link{plot.default}}}
}
\details{

  \code{\link{print.monomvn}} prints the \code{call} followed by a
  summary of the regression  method used at each iteration of the
  algorithm, also indicating how  many completely observed features
  (columns) there were in the data.
  For non-least-squares regressions (i.e., \pkg{plsr}, \pkg{lars}
  and \code{\link[MASS]{lm.ridge}} methods)
  and indication of the method used for selecting the
  number of components (i.e., \code{CV}, \code{LOO}, etc., or
  \code{none}) is provided

  \code{\link{summary.monomvn}} summarizes information about the
  number of zeros in the estimated covariance matrix \code{object$S}
  and its inverse.

  \code{\link{print.summary.monomvn}} calls \code{\link{print.monomvn}}
  on the \code{object} and then prints the result of
  \code{\link{summary.monomvn}}

  \code{\link{plot.summary.monomvn}} makes histograms of the number of
  zeros in the columns of \code{object$S} and its inverse
}
\value{
  \code{\link{summary.monomvn}} returns a \code{"summary.monomvn"}-class
  object, which is a list containing (a subset of) the items below.  The other
  functions do not return values.

  \item{obj}{ the \code{"monomvn"}-class \code{object} }
  \code{marg}{ the proportion of zeros in \code{object$S} }
  \item{S0}{ a vector containing the number of zeros in each column
    of \code{object$S} }
  \item{cond}{ if input \code{Si = TRUE} this field contains the
    proportion of zeros in the inverse of \code{object$S} }
  \item{Si0}{ if input \code{Si = TRUE} this field contains a
      vector with the number of zeros in each column of the inverse
      of \code{object$S} }
}
\author{ Robert B. Gramacy \email{bobby@statslab.cam.ac.uk} }
\references{
\url{http://www.statslab.cam.ac.uk/~bobby/monomvn.html}
}
\seealso{ \code{\link{monomvn}} }
\keyword{ methods }
\keyword{ hplot }
