/* Public numerical audit wrapper.  Include the official implementation so
   posterior algebra and proposal probabilities cannot drift from the sampler.
   H and h are row-major double arrays of length T*T and T respectively;
   cuts has MAXK slots (unused slots zero), and out has MAXNB*8 slots. */
#include "kernel.c"

static int probe_valid_partition(const int *cuts, int k, int minseg) {
    if (k < 0 || k > MAXK || minseg < 1 || minseg > T ||
        k + 1 > T / minseg || (k && !cuts)) return 0;
    int last = 0;
    for (int i = 0; i < k; ++i) {
        if (cuts[i] < last + minseg || cuts[i] > T - minseg) return 0;
        last = cuts[i];
    }
    return T - last >= minseg;
}

API double probe_weight(const double *H,const double *h,double r,int n,const int *cuts,int k,int m,double power) {
    if (!H || !h || n < 0 || !isfinite(r) || !isfinite(power) ||
        !probe_valid_partition(cuts, k, m)) return -INFINITY;
    double P[65*65],ph[65],A[MAXP*MAXP],mu[MAXP],rate,shape;
    prefix_sums(H,h,P,ph);
    return log_weight(P,ph,r,n,cuts,k,m,power,A,mu,&rate,&shape);
}

API int probe_neighbors(const double *H,const double *h,double r,int n,const int *cuts,int k,int kind,int km,int m,double power,double *out) {
    if (!H || !h || !out || !cuts || n < 0 || !isfinite(r) || !isfinite(power) ||
        kind < 0 || kind > 2 || km < k || km > MAXK || km < 0 ||
        !probe_valid_partition(cuts, k, m) || km + 1 > T / m) return -1;
    /* neighbors copies all MAXK slots during relocation.  Normalize the unused
       slots here rather than reading beyond the active cut count. */
    int padded[MAXK] = {0};
    for (int i = 0; i < k; ++i) padded[i] = cuts[i];
    double P[65*65],ph[65],q[MAXNB]; Neighbor nb[MAXNB];
    prefix_sums(H,h,P,ph);
    int count=neighbors(P,ph,r,n,padded,k,kind,km,m,power,nb);
    if(count) informed_probabilities(nb,count,q);
    for(int i=0;i<count;i++){
        out[i*8]=nb[i].k;
        for(int j=0;j<MAXK;j++) out[i*8+j+1]=nb[i].c[j];
        out[i*8+6]=nb[i].lw; out[i*8+7]=q[i];
    }
    return count;
}
