/* Separate bridge: kernel.c is included unchanged.  Structural state and RNG
   persist across changing latent observation weights. */
#include "kernel.c"

API int residual_init(int initk, uint64_t seed, int *state, uint64_t *rngstate) {
    if (initk < 0 || initk > 4) return 2;
    state[0] = initk;
    if (!initialise_cuts(initk, 8, state + 1)) return 3;
    *rngstate = seed ? seed : 1;
    return 0;
}

API int residual_step(const double *H, const double *h, double yqy, int n,
                      int *state, uint64_t *rngstate, double *out, long long *stats) {
    int k = state[0], *cuts = state + 1;
    RNG rng = {*rngstate};
    double P[65 * 65], ph[65], A[MAXP * MAXP], mu[MAXP], rate, shape;
    prefix_sums(H, h, P, ph);
    /* Crucial: H/h/yqy change after each latent-scale update. */
    double lw = log_weight(P, ph, yqy, n, cuts, k, 8, 1.0, A, mu, &rate, &shape);
    if (!isfinite(lw)) return 4;
    for (int round = 0; round < 2; ++round) {
        int kind = k == 0 ? 0 : (k == 4 ? 1 : (uniform01(&rng) < .5 ? 0 : 1));
        attempt(P, ph, yqy, n, kind, 4, 8, 1.0, &rng, &k, cuts, &lw, stats);
        for (int j = 0; j < 2; ++j) if (k)
            attempt(P, ph, yqy, n, 2, 4, 8, 1.0, &rng, &k, cuts, &lw, stats);
    }
    log_weight(P, ph, yqy, n, cuts, k, 8, 1.0, A, mu, &rate, &shape);
    if (!(rate > 0)) return 5;
    double L[MAXP * MAXP]; memcpy(L, A, sizeof(A));
    if (!cholesky(L, k + 1)) return 6;
    double s2 = rate / gamma_draw(&rng, shape), z[MAXP], noise[MAXP] = {0};
    for (int j = 0; j <= k; ++j) z[j] = normal01(&rng);
    /* Solve L' noise = z, giving covariance A^-1 without a second factor. */
    for (int i = k; i >= 0; --i) {
        double v = z[i];
        for (int j = i + 1; j <= k; ++j) v -= L[j * MAXP + i] * noise[j];
        noise[i] = v / L[i * MAXP + i];
    }
    out[0] = s2;
    for (int t = 1; t <= T; ++t) {
        int j = 0; while (j < k && t > cuts[j]) ++j;
        out[t] = mu[j] + sqrt(s2) * noise[j];
    }
    state[0] = k; *rngstate = rng.s;
    return 0;
}
