## print.monomvn
##
## generic print method for monomvn class objects,
## summarizing the results of a monomvn call

`print.monomvn` <-
function(x, ...)
  {
    cat("\nCall:\n")
    print(x$call)

    cat("\nMethods used:\n")
    um <- sort(unique(x$methods))
    for(u in um) {
      m <- x$methods == u
      cat(sum(m), "\t", u, sep="")
      if(u != "complete" && u != "lsr") {
        r <- range(x$ncomp[m])
        cat(paste(", ncomp range: [", r[1], ",", r[2], "]", sep=""))
      }
      cat("\n")
    }

    cat("\n")
  }
