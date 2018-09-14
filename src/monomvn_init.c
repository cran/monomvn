#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .C calls */
extern void adjust_elist_R(void *, void *, void *, void *, void *);
extern void blasso_cleanup();
extern void blasso_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void bmonomvn_cleanup();
extern void bmonomvn_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void get_regress_R(void *, void *, void *, void *, void *, void *, void *, void *);
extern void Igamma_inv_R(void *, void *, void *, void *, void *);
extern void mvnpdf_log_R(void *, void *, void *, void *, void *);
extern void rgig_R(void *, void *, void *, void *, void *);

static const R_CMethodDef CEntries[] = {
    {"adjust_elist_R",   (DL_FUNC) &adjust_elist_R,    5},
    {"blasso_cleanup",   (DL_FUNC) &blasso_cleanup,    0},
    {"blasso_R",         (DL_FUNC) &blasso_R,         40},
    {"bmonomvn_cleanup", (DL_FUNC) &bmonomvn_cleanup,  0},
    {"bmonomvn_R",       (DL_FUNC) &bmonomvn_R,       59},
    {"get_regress_R",    (DL_FUNC) &get_regress_R,     8},
    {"Igamma_inv_R",     (DL_FUNC) &Igamma_inv_R,      5},
    {"mvnpdf_log_R",     (DL_FUNC) &mvnpdf_log_R,      5},
    {"rgig_R",           (DL_FUNC) &rgig_R,            5},
    {NULL, NULL, 0}
};

void R_init_monomvn(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
