#******************************************************************************* 
#
# Estimation for Multivariate Normal Data with Monotone Missingness
# Copyright (C) 2007, University of Cambridge
# 
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Questions? Contact Robert B. Gramacy (bobby@statslab.cam.ac.uk)
#
#*******************************************************************************


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

  posdef <- function(S)
    {
      if(class(try(chol(S), silent=TRUE)) == "try-error" ||
         class(try(solve(S), silent=TRUE)) == "try-error") return(FALSE)
      else return(TRUE)
    }
  
  if(!posdef(S)) {
    
    ## check that library(accuracy) can be loaded
    if(require(accuracy, quietly=TRUE) == FALSE) {
      warning(paste(name, "not pos-def, install library(accuracy) for approx"))
      ## return(S)
    }

    ## print something to explain what is going on
    if(!quiet) warning(paste(name, "is not pos-def, using nearest approx"))

    ## make the approximation
    S.sechol <- try(sechol(S), silent=TRUE)
    if(class(S.sechol) != "try-error")
      S.approx <- t(S.sechol) %*% S.sechol

    ## see if the approximation worked.  If not, then just add a little
    ## to the diagonal of the original matrix and move on
    if(!posdef(S.approx)) {
      S.approx <- S
      eps <- 1
      cum <- 0
      while(1) {
        ## diag(S.approx) <- diag(S.approx) + .Machine$double.eps
        diag(S.approx) <- diag(S.approx) + eps
        cum <- cum + eps
        ## print(c(eps, cum))
        if(posdef(S.approx)) {
          if(eps > 2*.Machine$double.eps) {
            diag(S.approx) <- diag(S.approx) - eps
            cum <- cum - eps
            eps <- eps/2
          } else break;
        }
      }
     }
    return(S.approx)
    
  } else return(S)
}

