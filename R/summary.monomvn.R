## summary.monomvn
##
## generic summary method for monomvn class objects,
## records the percentage of marginal (and conditional
## when S=TRUE) columns is printed by analyising S (and Si)

'summary.monomvn' <-
function(object, Si=FALSE, ...)
  {
    rl <- list(obj=object)
    class(rl) <- "summary.monomvn"
    
    ## summary information about the zeros of
    ## the S matrix

    ## pairwise marginally uncorrelated
    rl$marg <-  sum(object$S == 0)/prod(dim(object$S))
    rl$S0 <- apply(object$S, 2, function(x) { sum(x == 0) })
      
    ## conditionally marginally uncorrelated
    if(Si) {
      Si <- solve(object$S)
      rl$cond <- sum(Si == 0)/prod(dim(Si))
      rl$Si0 <- apply(Si, 2, function(x) { sum(x == 0) })
    } 
    
    ## print it or return it
    rl
  }


## print.summary.monomvn
##
## print the results of the summary method after first
## calling the print method on the monomvn object.  

'print.summary.monomvn' <-
  function(x, ...)
{
  ## print the monomvn object
  print(x$obj, ...)

  ## count the extra number of things printed
  p <- 0
  
  ## print the marginally uncorrelated percentage
  if(!is.null(x$marg)) {
    cat(signif(100*x$marg, 3), "% of S is zero", sep="")
    cat(" (pairwise marginally uncorrelated [MUc])\n", sep="")
    m <- sum(x$S0 == length(x$S0) - 1)
    cat("\t", m, " cols (of ",  length(x$S0), " [",
        signif(100*m/length(x$S0), 3),
        "%]) are MI & CI of all others\n", sep="")
    p <- p + 1
  }

  ## print the conditionally uncorrellated percentage
  if(!is.null(x$cond)) {
    cat(signif(100*x$cond, 3), "% of inv(S) is zero", sep="")
    cat(" (pairwise conditionally uncorrelated [CI])\n", sep="")
    p <- p + 1
  }

  ## add another newline if we added anything to print.monomvn
  if(p > 0) cat("\n")
}


## plot.summary.monomvn:
##
## make historgrams of the number of zeros in the S matrix
## (and possibly inv(S) matrix) contained in a summary.monomvn
## object 

'plot.summary.monomvn' <-
  function(x, gt0=FALSE, main=NULL, xlab="number of zeros", ...)
{

  ## check if there is anything to plot
  if(all(x$S0 == 0)) {
    cat("S has no zero entries, so there is nothing to plot\n")
    return
  }
  
  ## count the number of things we've plotted
  p <- 0

  ## calculate the dimensions of the plot
  if(!is.null(x$S0) && !is.null(x$Si0))
    par(mfrow=c(1,2))

  ## agument main argument
  smain <- paste(main, "# of zero entries per column", sep="")
  
  ## plot a histogram of the number the zeros for each
  ## asset in S, marking marginal uncorrelation
  if(!is.null(x$S0)) {
    main <- paste(smain, "in S")
    if(gt0) { i <- x$S0 > 0; main <- paste(main, "[>0]") }
    else i <- rep(TRUE, length(x$S0))
    hist(x$S0[i], main=main, xlab=xlab, ...)
    p <- p + 1
  }

  ## plot a histogram of the number of zeros for each
  ## asset in Si, marking conditional uncorrelation
  if(!is.null(x$Si0)) {
    main <- paste(smain, "in inv(S)")
    if(gt0) { i <- x$Si0 > 0; main <- paste(main, "[>0]") }
    else i <- rep(TRUE, length(x$Si0))
    hist(x$Si0[i], main=main, xlab=xlab, ...)
    p <- p + 1
  }
  
  if(p == 0) warning("nothing to plot")
}
