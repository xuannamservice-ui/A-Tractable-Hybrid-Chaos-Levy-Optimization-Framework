"""
Evaluate eq (22) -- the lambda_j / C_j convolved series -- at K=10, tractably.

The branch density is  f_h(h) = sum_i d_i h^{mu_i - 1}  with
    (mu, d) = (xi^2, D), (beta+k, a_k(a,b)), (alpha+k, a_k(b,a)),  k = 0..K
Its Laplace transform is  P(s) = sum_i d_i Gamma(mu_i) s^{-mu_i}, so the MN-fold
convolution has transform P(s)^MN and therefore

    f_H(H) = sum_j C_j H^{lambda_j - 1},   C_j = B_j / Gamma(lambda_j)
    P(s)^MN = sum_j B_j s^{-lambda_j}
    ABER_sys = sum_j C_j * C(lambda_j, gbar/MN)

Naive multinomial expansion is hopeless: 23 branch terms to the 16th power is
C(38,22) = 2.22e10 partitions.  But the exponents live on a lattice.  Writing

    P = u + x^beta * Bp(x) + x^alpha * Ap(x),
    u = D*Gamma(xi^2) x^{xi^2},  Bp(x) = sum_k a_k(a,b) Gamma(beta+k) x^k, etc.

the trinomial theorem gives

    P^MN = sum_{nD+nB+nA=MN} MN!/(nD! nB! nA!) * u^{nD}
             * x^{nB*beta} Bp^{nB} * x^{nA*alpha} Ap^{nA}

so every convolved exponent is  lambda = nD*xi^2 + nB*beta + nA*alpha + S,
with S the total k-index carried by the two ordinary polynomial powers.
That is 153 (nD,nB,nA) triples times S <= MN*K -- a few tens of thousands of
terms, all reachable.
"""
import mpmath as mp
from rtodt import REGIMES, A0_for, a_k, D_coef, C_moment, db

mp.mp.dps = 260          # the a_k alternate in sign over ~26 decades

XI = mp.mpf("1.967")
SIGMA = mp.mpf("0.05")
MN = 16
K = 10


def polypow(coeffs, n, cache):
    """Ordinary polynomial power, coefficients indexed by degree."""
    if n in cache:
        return cache[n]
    if n == 0:
        r = [mp.mpf(1)]
    else:
        prev = polypow(coeffs, n - 1, cache)
        r = [mp.mpf(0)] * (len(prev) + len(coeffs) - 1)
        for i, a in enumerate(prev):
            if a == 0:
                continue
            for j, b in enumerate(coeffs):
                if b != 0:
                    r[i + j] += a * b
    cache[n] = r
    return r


def eq22(regime, snr_db_list):
    A, B = REGIMES[regime]
    A0 = A0_for(XI, SIGMA)
    x2 = XI ** 2

    # branch-transform pieces
    uD = D_coef(A, B, XI, A0) * mp.gamma(x2)
    Bp = [a_k(A, B, XI, A0, k) * mp.gamma(B + k) for k in range(K + 1)]
    Ap = [a_k(B, A, XI, A0, k) * mp.gamma(A + k) for k in range(K + 1)]

    cB, cA = {}, {}
    fact = [mp.factorial(i) for i in range(MN + 1)]

    # accumulate B_j by exponent lambda
    acc = {}
    for nD in range(MN + 1):
        for nB in range(MN + 1 - nD):
            nA = MN - nD - nB
            multi = fact[MN] / (fact[nD] * fact[nB] * fact[nA])
            uterm = uD ** nD if nD else mp.mpf(1)
            pB = polypow(Bp, nB, cB)
            pA = polypow(Ap, nA, cA)
            base = multi * uterm
            lam0 = nD * x2 + nB * B + nA * A
            for i, bi in enumerate(pB):
                if bi == 0:
                    continue
                bb = base * bi
                for j, aj in enumerate(pA):
                    if aj == 0:
                        continue
                    lam = lam0 + i + j
                    key = mp.nstr(lam, 25)
                    if key in acc:
                        acc[key] = (acc[key][0], acc[key][1] + bb * aj)
                    else:
                        acc[key] = (lam, bb * aj)

    print("  %s: %d distinct convolved exponents lambda_j" % (regime, len(acc)))

    out = {}
    for g in snr_db_list:
        gb = db(g) / MN
        tot = mp.mpf(0)
        for lam, Bj in acc.values():
            tot += (Bj / mp.gamma(lam)) * C_moment(lam, gb)
        out[g] = tot
    return out


def reference(snr_db_list):
    """Independent reference: build f_h by quadrature over the pointing law and
    convolve it MN times by FFT, then integrate Q against it.

    This shares no code path with eq22() above -- no a_k, no lambda_j, no
    trinomial expansion -- so agreement between the two is evidence about the
    series, not a transcription check.

    hmax / nh follow generate.py: the branch mean is A_0 xi^2/(xi^2+1) < A_0,
    so a grid to 40*A_0 covers the density to well past the point where Q()
    has stopped contributing, and 60000 points resolve its rise near zero.
    The recovered mass of f_H is printed so the reader can see the quadrature
    is converged rather than take it on trust.
    """
    from egc_system import aber_system, f_h_exact
    A, B = REGIMES["strong"]
    A0 = A0_for(XI, SIGMA)
    out = {}
    for g in snr_db_list:
        v, mass, _, floor = aber_system(f_h_exact, 40 * float(A0), 60000,
                                        float(db(g)), float(A), float(B),
                                        float(XI), float(A0), "exact",
                                        return_floor=True)
        out[g] = (v, mass, floor)
    return out


if __name__ == "__main__":
    SNRS = [20, 28, 32, 40]
    # Values recorded from a previous run of reference() below, kept only so a
    # reader can spot a regression at a glance. They are NOT used to compute
    # anything, and the "reference" column is recomputed live on every run.
    FROZEN = {20: 4.510e-3, 28: 2.240e-5, 32: 6.200e-7, 40: 9.040e-11}
    print("eq (22) via the lambda_j/C_j convolved series, K=%d, MN=%d" % (K, MN))
    print("xi=%s, sigma_s=%s, strong turbulence\n" % (XI, SIGMA))
    res = eq22("strong", SNRS)
    print("  computing the independent 16-fold-convolution reference "
          "(about 20 s per SNR)...")
    ref = reference(SNRS)
    print()
    print("  %-8s %-22s %-16s %-10s %-10s %-11s %s"
          % ("SNR", "eq (22)", "reference", "rel. diff", "FFT floor",
             "f_H mass", "prev. run"))
    for g in SNRS:
        v = res[g]
        r, mass, floor = ref[g]
        rel = (float(v) - r) / r * 100 if r else float("nan")
        mark = "" if abs(r) > 10 * floor else "  <-- AT THE FLOOR"
        print("  %-8s %-22s %-16.4e %-10s %-10.2e %-11.6f %.3e%s"
              % ("%d dB" % g, mp.nstr(v, 8), r, "%+.3f%%" % rel, floor, mass,
                 FROZEN[g], mark))
    print()
    print("  The two columns are independent computations of the same system")
    print("  ABER; the reference carries its own quadrature error, which the")
    print("  recovered mass of f_H bounds, and a double-precision FFT round-off")
    print("  floor, printed so a row that has sunk into it is visible rather")
    print("  than merely small.")
