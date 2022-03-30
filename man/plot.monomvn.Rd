\name{plot.monomvn}
\alias{plot.monomvn}
\title{ Plotting bmonomvn output }
\description{
  Functions for visualizing the output from \code{\link{bmonomvn}},
  particularly the posterior standard deviation estimates of the mean
  vector and covariance matrix, and samples from the solution to a
  Quadratic Program
}
\usage{
\method{plot}{monomvn}(x, which=c("mu", "S", "Snz", "Sinz", "QP"),
     xaxis=c("numna", "index"), main=NULL, uselog=FALSE, ...)
}
\arguments{
  \item{x}{ a \code{"monomvn"}-class object that must be named \code{x}
    for generic plotting }
  \item{which}{ determines the parameter whose standard deviation
    to be visualized: the mean vector (\code{"mu"} for \code{sqrt($mu.var)});
    the covariance \code{matrix} (\code{"S"} for \code{sqrt($S.var)}),
    or \code{"S{i}nz"} for \code{sqrt($S{i}.nz)}, which both result in
    an \code{\link{image}} plot; or the distribution of solutions
    \code{$W} to a Quadratic Program that may be obtained by
    supplying \code{QP = TRUE} as input to \code{\link{bmonomvn}} }
  \item{xaxis}{ indicates how x-axis (or x- and y-axis in the case
    of \code{which = "S" || "S{i}nz"}) should be displayed.  The default option
    \code{"numna"} shows the (ordered) number of missing entries
    (\code{NA}s) in the corresponding column, whereas \code{"index"}
    simply uses the column index; see details below }
  \item{main}{ optional text to be added to the main title of the plots;
    the default of \code{NULL} causes the automatic generation of a
    title }
  \item{uselog}{ a logical which, when \code{TRUE}, causes the log
    of the standard deviation to be plotted instead }
  \item{...}{ passed to \code{\link{plot.default}}}
}
\details{
  Currently, this function only provides a visualization of the
  posterior standard deviation estimates of the parameters, and
  the distributions of samples from the posterior of the solution
  to a specified Quadratic Program.  Therefore
  it only works on the output from \code{\link{bmonomvn}}

  All types of visualization (specified by \code{which}) are presented
  in the order of the number of missing entries in the columns of the
  data passed as input to  \code{\link{bmonomvn}}.
  In the case of \code{which = "mu"}
  this means that y-values are presented in the order \code{x$o}, where
  the x-axis is either \code{1:length(x$o)} in the case of
  \code{xaxis = "index"}, or \code{x$na[x$o]} in the case of \code{xaxis
    = "numna"}.  When \code{which = "S"} is given the resulting
  \code{\link{image}} plot is likewise ordered by \code{x$o} where the
  x- and y-axis are as above, except that in the case
  where \code{xaxis = "numna"} the repeated counts of \code{NA}s are
  are adjusted by small increments so that x and y arguments to
  \code{\link{image}} are distinct.  Since a \code{\link{boxplot}} is
  used when \code{which = "QP"} it may be that \code{xaxis = "index"}
  is preferred
}
\value{
  The only output of this function is beautiful plots
}
\author{ Robert B. Gramacy \email{rbg@vt.edu} }
\references{
\url{https://bobby.gramacy.com/r_packages/monomvn/}
}
\seealso{ \code{\link{bmonomvn}}, \code{\link{print.monomvn}},
           \code{\link{summary.monomvn}} }
\keyword{ hplot }
