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


my.mvnpdf <- function(x, mu, S, log=FALSE)
  {
    r <- .C("mvnpdf_log_R",
       x = as.double(x),
       mu = as.double(mu),
       S = as.double(S),
       n = as.integer(length(x)),
       result =  double(1)
       ,PACKAGE = "monomvn")

    if(log) return(r$result)
    else return(exp(r$result))
  }
    
