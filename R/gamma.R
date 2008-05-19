my.Igamma.inv <- function(a, y, lower=FALSE, log=FALSE)
  {
    ## call the C routine
    r <- .C("Igamma_inv_R",
            a = as.double(a),
            y = as.double(y),
            lower = as.integer(lower),
            log = as.integer(log),
            result = double(1),
            PACKAGE = "monomvn")

    return(r$result)
  }
