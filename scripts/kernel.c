#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

#define T 64
#define MAXK 5
#define MAXP 6
#define MAXNB 512
#define WIDTH (1 + MAXK + T)

/* The runner passes collapsed sufficient statistics for the T=64 trial effects:
   H=X'QX (64 by 64), h=X'Qy (64), and yqy=y'Qy.  Each stored draw occupies
   [K, MAXK cut slots, T fitted trial effects]. */
typedef struct { uint64_t s; } RNG;
typedef struct { int k, c[MAXK]; double lw; } Neighbor;

/* Deterministic xorshift64* stream.  All proposal and conditional draws use
   this one state, so a nonzero seed fully determines a chain. */
static uint64_t next_u64(RNG *r) {
    uint64_t x = r->s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27; r->s = x;
    return x * UINT64_C(2685821657736338717);
}
static double uniform01(RNG *r) { return ((next_u64(r) >> 11) + .5) * 0x1p-53; }
static double normal01(RNG *r) {
    return sqrt(-2.0 * log(uniform01(r))) * cos(6.283185307179586 * uniform01(r));
}
static double gamma_draw(RNG *r, double a) {
    /* Marsaglia--Tsang Gamma(shape=a, scale=1), with the standard a<1 transform. */
    if (a < 1.0) return gamma_draw(r, a + 1.0) * pow(uniform01(r), 1.0 / a);
    double d = a - 1.0 / 3.0, c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x = normal01(r), v = 1.0 + c * x;
        if (v <= 0.0) continue;
        v *= v * v;
        double u = uniform01(r);
        if (u < 1.0 - .0331 * x * x * x * x ||
            log(u) < .5 * x * x + d * (1.0 - v + log(v))) return d * v;
    }
}

static double move_type_probability(int k, int birth, int kmax) {
    /* Birth/death selection is forced at the feasible K bounds and balanced inside. */
    if (k == 0) return birth ? 1.0 : 0.0;
    if (k == kmax) return birth ? 0.0 : 1.0;
    return .5;
}
static double log_partition_count(int k, int minseg) {
    /* Number of ordered k-cut partitions satisfying the minimum segment length. */
    if (k == 0) return 0.0;
    int top = T - minseg * (k + 1) + k;
    if (top < k) return INFINITY;
    return lgamma((double)top + 1.0) - lgamma((double)k + 1.0) - lgamma((double)(top - k) + 1.0);
}
static int cholesky(double *A, int p) {
    for (int i = 0; i < p; ++i) for (int j = 0; j <= i; ++j) {
        double s = A[i * MAXP + j];
        for (int q = 0; q < j; ++q) s -= A[i * MAXP + q] * A[j * MAXP + q];
        if (i == j) { if (s <= 0.0) return 0; A[i * MAXP + j] = sqrt(s); }
        else A[i * MAXP + j] = s / A[j * MAXP + j];
    }
    return 1;
}
static void solve_cholesky(const double *L, const double *b, double *x, int p) {
    double z[MAXP];
    for (int i = 0; i < p; ++i) {
        double s = b[i]; for (int j = 0; j < i; ++j) s -= L[i * MAXP + j] * z[j];
        z[i] = s / L[i * MAXP + i];
    }
    for (int i = p - 1; i >= 0; --i) {
        double s = z[i]; for (int j = i + 1; j < p; ++j) s -= L[j * MAXP + i] * x[j];
        x[i] = s / L[i * MAXP + i];
    }
}
static void prefix_sums(const double *H, const double *h, double *P, double *ph) {
    /* Summed-area tables make every segment-pair block sum O(1). */
    memset(P, 0, 65 * 65 * sizeof(double)); ph[0] = 0.0;
    for (int i = 0; i < T; ++i) {
        ph[i + 1] = ph[i] + h[i];
        for (int j = 0; j < T; ++j)
            P[(i + 1) * 65 + j + 1] = H[i * T + j] + P[i * 65 + j + 1] + P[(i + 1) * 65 + j] - P[i * 65 + j];
    }
}
static double log_weight(const double *P, const double *ph, double yqy, int n,
                         const int *cuts, int k, int minseg, double prior_power,
                         double *A, double *mu, double *rate, double *shape) {
    /* Integrate segment levels under delta_j | sigma2 ~ N(0, sigma2/4)
       and sigma2 ~ IG(shape=3, rate=2); include the K!^{-prior_power}
       and uniform-valid-partition prior terms in the collapsed log weight. */
    int edges[MAXP + 1], p = k + 1; edges[0] = 0;
    for (int i = 0; i < k; ++i) edges[i + 1] = cuts[i];
    edges[p] = T;
    double g[MAXP] = {0}, L[MAXP * MAXP] = {0}; memset(A, 0, MAXP * MAXP * sizeof(double));
    for (int i = 0; i < p; ++i) {
        g[i] = ph[edges[i + 1]] - ph[edges[i]];
        for (int j = 0; j < p; ++j)
            A[i * MAXP + j] = P[edges[i + 1] * 65 + edges[j + 1]] - P[edges[i] * 65 + edges[j + 1]]
                                  - P[edges[i + 1] * 65 + edges[j]] + P[edges[i] * 65 + edges[j]];
        A[i * MAXP + i] += 4.0;
    }
    memcpy(L, A, sizeof(L)); if (!cholesky(L, p)) return NAN;
    memset(mu, 0, MAXP * sizeof(double)); solve_cholesky(L, g, mu, p);
    *rate = 2.0 + .5 * yqy; for (int i = 0; i < p; ++i) *rate -= .5 * g[i] * mu[i];
    *shape = 3.0 + n / 2.0;
    double logdet_half = 0.0; for (int i = 0; i < p; ++i) logdet_half += log(L[i * MAXP + i]);
    return -prior_power * lgamma((double)k + 1.0) - log_partition_count(k, minseg)
           + .5 * p * log(4.0) - logdet_half - (*shape) * log(*rate);
}
static int neighbors(const double *P, const double *ph, double yqy, int n, const int *cuts,
                     int k, int kind, int kmax, int minseg, double prior_power, Neighbor *out) {
    /* kind: 0=birth all legal within-segment cuts; 1=death each existing cut;
       2=move one cut to each other legal position.  Every candidate gets its
       own collapsed target weight. */
    int count = 0;
    if (kind == 0 && k < kmax) {
        int lo = 0;
        for (int seg = 0; seg <= k; ++seg) {
            int hi = seg < k ? cuts[seg] : T;
            for (int t = lo + minseg; t <= hi - minseg; ++t) {
                Neighbor *x = &out[count++]; x->k = k + 1; memset(x->c, 0, sizeof(x->c));
                int q = 0; for (int i = 0; i < k; ++i) { if (q == seg) x->c[q++] = t; x->c[q++] = cuts[i]; }
                if (q == seg) x->c[q++] = t;
                double A[MAXP * MAXP], mu[MAXP], rate, shape;
                x->lw = log_weight(P, ph, yqy, n, x->c, x->k, minseg, prior_power, A, mu, &rate, &shape);
            }
            lo = hi;
        }
    } else if (kind == 1 && k > 0) {
        for (int drop = 0; drop < k; ++drop) {
            Neighbor *x = &out[count++]; x->k = k - 1; memset(x->c, 0, sizeof(x->c));
            int q = 0; for (int i = 0; i < k; ++i) if (i != drop) x->c[q++] = cuts[i];
            double A[MAXP * MAXP], mu[MAXP], rate, shape;
            x->lw = log_weight(P, ph, yqy, n, x->c, x->k, minseg, prior_power, A, mu, &rate, &shape);
        }
    } else if (kind == 2 && k > 0) {
        for (int j = 0; j < k; ++j) {
            int lo = (j ? cuts[j - 1] : 0) + minseg;
            int hi = (j + 1 < k ? cuts[j + 1] : T) - minseg;
            for (int t = lo; t <= hi; ++t) if (t != cuts[j]) {
                Neighbor *x = &out[count++]; x->k = k; memcpy(x->c, cuts, sizeof(x->c)); x->c[j] = t;
                double A[MAXP * MAXP], mu[MAXP], rate, shape;
                x->lw = log_weight(P, ph, yqy, n, x->c, k, minseg, prior_power, A, mu, &rate, &shape);
            }
        }
    }
    return count;
}
static void informed_probabilities(Neighbor *x, int n, double *q) {
    /* Locally balanced proposal: 95% normalized sqrt-target weights plus a
       5% uniform component, which keeps every enumerated neighbor reachable. */
    double mx = -INFINITY, sum = 0.0;
    for (int i = 0; i < n; ++i) if (.5 * x[i].lw > mx) mx = .5 * x[i].lw;
    for (int i = 0; i < n; ++i) { q[i] = exp(.5 * x[i].lw - mx); sum += q[i]; }
    for (int i = 0; i < n; ++i) q[i] = .05 / n + .95 * q[i] / sum;
}
static int same_state(const Neighbor *x, int k, const int *cuts) {
    if (x->k != k) return 0;
    for (int i = 0; i < MAXK; ++i) if (x->c[i] != cuts[i]) return 0;
    return 1;
}
static int attempt(const double *P, const double *ph, double yqy, int n, int kind, int kmax,
                   int minseg, double prior_power, RNG *rng, int *k, int *cuts, double *lw, long long *stats) {
    /* Enumerate the reverse neighborhood at the proposed state.  The MH ratio
       includes both direction-selection probabilities and reverse proposal mass. */
    Neighbor nb[MAXNB], rev[MAXNB]; double q[MAXNB], qr[MAXNB];
    int nn = neighbors(P, ph, yqy, n, cuts, *k, kind, kmax, minseg, prior_power, nb); if (!nn) return 0;
    informed_probabilities(nb, nn, q); double u = uniform01(rng), cum = 0.0; int pick = nn - 1;
    for (int i = 0; i < nn; ++i) { cum += q[i]; if (u <= cum) { pick = i; break; } }
    int reverse_kind = kind == 0 ? 1 : (kind == 1 ? 0 : 2);
    int rn = neighbors(P, ph, yqy, n, nb[pick].c, nb[pick].k, reverse_kind, kmax, minseg, prior_power, rev);
    informed_probabilities(rev, rn, qr); double back = 0.0;
    for (int i = 0; i < rn; ++i) if (same_state(&rev[i], *k, cuts)) { back = qr[i]; break; }
    double tf = kind == 0 ? move_type_probability(*k, 1, kmax) : (kind == 1 ? move_type_probability(*k, 0, kmax) : 1.0);
    double tr = kind == 0 ? move_type_probability(nb[pick].k, 0, kmax) : (kind == 1 ? move_type_probability(nb[pick].k, 1, kmax) : 1.0);
    double log_alpha = nb[pick].lw - *lw + log(tr * back / (tf * q[pick])); stats[kind * 2]++;
    if (log(uniform01(rng)) < log_alpha) { *k = nb[pick].k; memcpy(cuts, nb[pick].c, sizeof(nb[pick].c)); *lw = nb[pick].lw; stats[kind * 2 + 1]++; return 1; }
    return 0;
}
static int initialise_cuts(int k, int minseg, int *cuts) {
    /* Start at evenly spaced legal cuts; this is initialization only, not a prior. */
    memset(cuts, 0, MAXK * sizeof(int));
    for (int j = 1; j <= k; ++j) cuts[j - 1] = (T * j) / (k + 1);
    for (int j = 0; j < k; ++j) {
        int lo = (j ? cuts[j - 1] : 0) + minseg;
        int hi = T - minseg * (k - j);
        if (cuts[j] < lo) cuts[j] = lo;
        if (cuts[j] > hi) cuts[j] = hi;
    }
    return k <= MAXK && minseg * (k + 1) <= T;
}

API int r61_formal_run(const double *H, const double *h, double yqy, int n, int iters, int burn,
                       uint64_t seed, int initk, int kmax, int minseg, double prior_power,
                       double *out, long long *stats) {
    /* Fixed bounds protect static workspaces and ensure every K<=kmax has a
       valid minseg-constrained partition.  Return codes are consumed by Python. */
    if (kmax < 0 || kmax > MAXK || minseg < 1 || minseg * (kmax + 1) > T ||
        initk < 0 || initk > kmax || burn < 0 || iters <= burn) return 2;
    double P[65 * 65], ph[65]; prefix_sums(H, h, P, ph); RNG rng = {seed ? seed : 1};
    int k = initk, cuts[MAXK]; if (!initialise_cuts(k, minseg, cuts)) return 3;
    double A[MAXP * MAXP], mu[MAXP], rate, shape;
    double lw = log_weight(P, ph, yqy, n, cuts, k, minseg, prior_power, A, mu, &rate, &shape);
    memset(stats, 0, 8 * sizeof(long long)); int kept = 0;
    for (int it = 0; it < iters; ++it) {
        /* Two birth/death rounds, each followed by two within-model moves. */
        for (int round = 0; round < 2; ++round) {
            int kind = k == 0 ? 0 : (k == kmax ? 1 : (uniform01(&rng) < .5 ? 0 : 1));
            attempt(P, ph, yqy, n, kind, kmax, minseg, prior_power, &rng, &k, cuts, &lw, stats);
            for (int j = 0; j < 2; ++j) if (k) attempt(P, ph, yqy, n, 2, kmax, minseg, prior_power, &rng, &k, cuts, &lw, stats);
        }
        lw = log_weight(P, ph, yqy, n, cuts, k, minseg, prior_power, A, mu, &rate, &shape);
        double inverse[MAXP * MAXP] = {0}, L[MAXP * MAXP] = {0};
        for (int col = 0; col < k + 1; ++col) {
            double unit[MAXP] = {0}, x[MAXP] = {0}, LA[MAXP * MAXP]; memcpy(LA, A, sizeof(A));
            if (!cholesky(LA, k + 1)) return 4;
            unit[col] = 1.0;
            solve_cholesky(LA, unit, x, k + 1);
            for (int i = 0; i < k + 1; ++i) inverse[i * MAXP + col] = x[i];
        }
        memcpy(L, inverse, sizeof(L)); if (!cholesky(L, k + 1)) return 5;
        /* Conditional order is sigma2 | partition,data first, then
           delta | sigma2,partition,data.  The inverse precision is factored
           before the Gaussian draw. */
        double s2 = rate / gamma_draw(&rng, shape), z[MAXP], delta[MAXP] = {0};
        for (int j = 0; j < k + 1; ++j) z[j] = normal01(&rng);
        for (int i = 0; i < k + 1; ++i) { double v = 0.0; for (int j = 0; j <= i; ++j) v += L[i * MAXP + j] * z[j]; delta[i] = mu[i] + sqrt(s2) * v; }
        if (it >= burn) {
            /* Retain only post-burn draws; expand piecewise segment levels to
               the 64 trial positions expected by the Python output schema. */
            double *row = out + WIDTH * kept++; row[0] = k;
            for (int j = 0; j < MAXK; ++j) row[1 + j] = cuts[j];
            for (int t = 1; t <= T; ++t) { int j = 0; while (j < k && t > cuts[j]) ++j; row[1 + MAXK + t - 1] = delta[j]; }
        }
    }
    return 0;
}
