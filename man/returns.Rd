\name{returns}
\alias{returns}
\alias{returns.test}
\alias{market}
\alias{market.test}
\docType{data}
\title{Financial Returns data from NYSE and AMEX}
\description{
Monthly returns of common domestic stocks traded on the NYSE and the
AMEX from April 1968 until 1998; also contains the return to the market
}
\usage{
data(returns)
data(returns.test)
data(market)
data(market.test)
}
\format{
  The returns provided are collected in a \code{data.frame} with
  1168 columns, and 360 rows in the case of \code{returns} and 12
  rows for \code{returns.test}. The columns are uniquely coded to
  identify the stock traded on NYSE or AMEX.  The market return
  is in two vectors \code{market} and \code{market.test}
  of length 360 and 12, respectively
}
\details{
  The columns contain monthly returns of common domestic stocks traded
  on the NYSE and the AMEX from April 1968 until 1998. \code{returns}
  contains returns up until 1997, whereas \code{returns.test} has the
  returns for 1997.  Both data sets have been cleaned in the following
  way.  All stocks have a share price greater than \$5 and a market
  capitalization greater than 20\% based on the size distribution of
  NYSE firms.  Stocks without completely observed return
  series in 1997 were also discarded.

  The market returns provided are essentially the monthly return on the
  S\&P500 during the same period, which is highly correlated with the
  raw monthly returns weighted by their market capitalization
}
\source{
  This data is a subset of that originally used by Chan, Karceski,
  and Lakonishok (1999), and subsequently by several others;
  see the references below.  We use it as part of the \pkg{monomvn}
  package as an example of a real world data set following a
  nearly monotone missingness pattern
}
\references{
  Louis K. Chan, Jason Karceski, and Josef Lakonishok (1999).
  \emph{On Portfolio Optimization: Forecasting Covariances and
    Choosing the Risk Model}. The Review of Financial Studies.
  \bold{12}(5), 937-974

  Ravi Jagannathan and Tongshu Ma (2003).
  \emph{Risk Reduction in Large Portfolios: Why Imposing the
    Wrong Constraints Helps}. Journal of Finance, American Finance
  Association. \bold{58}(4), 1641-1684

  Robert B. Gramacy, Joo Hee Lee, and Ricardo Silva (2008).
  \emph{On estimating covariances between many assets with histories 
  of highly variable length}. \cr Preprint available on arXiv:0710.5837:
  \url{http://arxiv.org/abs/0710.5837}

  \url{http://www.statslab.cam.ac.uk/~ bobby/monomvn.html}
}

\seealso{
  \code{\link{monomvn}}, \code{\link{bmonomvn}}
  }

\examples{
data(returns)

## investigate the monotone missingness pattern
returns.na <- is.na(returns)
image(1:ncol(returns), 1:nrow(returns), t(returns.na))

## for a portfolio balancing exercise, see
## the example in the bmonomvn help file
}
\keyword{datasets}
