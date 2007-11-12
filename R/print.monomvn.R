## print.monomvn
##
## generic print method for monomvn class objects,
## summarizing the results of a monomvn call

`print.monomvn` <-
function(x, ...)
  {
    cat("\nCall:\n")
    print(x$call)

    cat("\nMethods used (p=", x$p, "):\n", sep="")
    um <- sort(unique(x$methods))
    for(u in um) {
      m <- x$methods == u
      cat(sum(m), "\t", u, sep="")
      if(u != "complete" && u != "lsr") {
        r <- range(x$ncomp[m])
        ncomp <- "ncomp"
        if(u == "ridge") ncomp <- "lambda"
        cat(paste(", ", ncomp, " range: [",
                  signif(r[1],5), ",", signif(r[2],5), "]", sep=""))
      }
      cat("\n")
    }

    cat("\n")
  }
