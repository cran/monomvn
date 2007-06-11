## posdef.approx
##
## uses sechol in library(accuracy) to find the nearest
## positive definite matrix to S

`posdef.approx` <-
function(S, name="S", quiet=FALSE)
{
  ## sanity check
  if(!is.matrix(S) || nrow(S) != ncol(S))
    stop(paste(name, "needs to be a symmetric matrix"))
  
  if(class(try(chol(S), silent=TRUE)) == "try-error") {

    ## check that library(accuracy) can be loaded
    if(require(accuracy, quietly=TRUE) == FALSE) {
      warning(paste(name, "is not pos-def, install library(accuracy) for nearest approx"))
      return(S)
    } else if(!quiet) warning(paste(name, "is not pos-def, using nearest approx"))

    ## make the approximation
    S.sechol <- sechol(S)
    S <- t(S.sechol) %*% S.sechol
  }

  return(S)
}

