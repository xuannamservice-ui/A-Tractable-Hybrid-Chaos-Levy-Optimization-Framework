"""
Re-implementation of the RT-ODT power-series ABER emulator from access.tex,
built ONLY from the equations printed in the manuscript:

  eq:hp_def     A_0 = [erf(v)]^2,  v = sqrt(pi/2) a / w_z
                w_zeq^2 = w_z^2 * sqrt(pi) * erf(v) / (2 v e^{-v^2})
                xi = w_zeq / (2 sigma_s)
  eq:ak_formula a_k(A,B,xi) = xi^2/(G(A)G(B)) * (-1)^k (A*B)^{B+k} G(A-B-k)
                              / ( k! (xi^2-B-k) A_0^{B+k} )
  eq:D_formula  D(A,B,xi)   = xi^2 (A*B)^{xi^2} G(A-xi^2) G(B-xi^2)
                              / ( A_0^{xi^2} G(A) G(B) )
  eq:Cs_closed  C(s,gbar)   = G((s+1)/2) / (2 s sqrt(pi)) * (2/gbar)^{s/2}
  eq:aber_emulator  Pe = D*C(xi^2) + sum_k [ a_k(A,B)C(B+k) + a_k(B,A)C(A+k) ]

All heavy arithmetic in mpmath at high precision, because the series is
sign-alternating with ~26 decades of dynamic range.
"""
import mpmath as mp

mp.mp.dps = 90

APERTURE = mp.mpf("0.05")          # a, metres (line 146)
NODES = [mp.mpf(x) for x in
         ["0.500", "0.628", "0.789", "0.992", "1.266",
          "1.548", "1.967", "2.511", "3.104", "3.912", "4.888"]]
REGIMES = {"weak": (mp.mpf("4.2"), mp.mpf("3.0")),
           "moderate": (mp.mpf("2.1"), mp.mpf("1.5")),
           "strong": (mp.mpf("1.2"), mp.mpf("1.1"))}
SIGMAS = [mp.mpf(s) for s in ["0.05", "0.1", "0.2", "0.3"]]


# ---------------------------------------------------------------- geometry
def v_of(wz, a=APERTURE):
    return mp.sqrt(mp.pi / 2) * a / wz


def A0_of(wz, a=APERTURE):
    return mp.erf(v_of(wz, a)) ** 2


def wzeq_of(wz, a=APERTURE):
    v = v_of(wz, a)
    return mp.sqrt(wz**2 * mp.sqrt(mp.pi) * mp.erf(v) / (2 * v * mp.e**(-v**2)))


def wzeq_min(a=APERTURE):
    """Farid-Hranilovic w_zeq is non-monotonic in w_z; find its minimum."""
    f = lambda w: wzeq_of(mp.mpf(w), a)
    lo, hi = mp.mpf("0.005"), mp.mpf("0.5")
    # golden-section on a unimodal-in-log region
    gr = (mp.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    for _ in range(400):
        if f(c) < f(d):
            hi = d
        else:
            lo = c
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    w = (lo + hi) / 2
    return w, f(w)


_WMIN, _WEQMIN = wzeq_min()


def wz_for_xi(xi, sigma_s, a=APERTURE):
    """Invert xi = w_zeq/(2 sigma_s) on the UPPER branch (w_z > argmin),
    which is the physically meaningful beam-broadening branch on which
    A_0 decreases with xi (manuscript line 258)."""
    target = 2 * sigma_s * xi
    if target <= _WEQMIN:
        return None
    lo, hi = _WMIN, mp.mpf("50")
    for _ in range(400):
        mid = (lo + hi) / 2
        if wzeq_of(mid, a) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def A0_for(xi, sigma_s):
    wz = wz_for_xi(xi, sigma_s)
    return None if wz is None else A0_of(wz)


# ---------------------------------------------------------- series pieces
def a_k(A, B, xi, A0, k):
    x2 = xi**2
    num = (-1)**k * (A * B)**(B + k) * mp.gamma(A - B - k)
    den = mp.factorial(k) * (x2 - B - k) * A0**(B + k)
    return x2 / (mp.gamma(A) * mp.gamma(B)) * num / den


def D_coef(A, B, xi, A0):
    x2 = xi**2
    return (x2 * (A * B)**x2 * mp.gamma(A - x2) * mp.gamma(B - x2)
            / (A0**x2 * mp.gamma(A) * mp.gamma(B)))


def C_moment(s, gbar):
    return mp.gamma((s + 1) / 2) / (2 * s * mp.sqrt(mp.pi)) * (2 / gbar)**(s / 2)


def Pe_series(A, B, xi, A0, gbar, K):
    """eq:aber_emulator, evaluated exactly (no interpolation)."""
    x2 = xi**2
    tot = D_coef(A, B, xi, A0) * C_moment(x2, gbar)
    for k in range(K + 1):
        tot += a_k(A, B, xi, A0, k) * C_moment(B + k, gbar)
        tot += a_k(B, A, xi, A0, k) * C_moment(A + k, gbar)
    return tot


def max_abs_ak(A, B, xi, A0, K):
    vals = []
    for k in range(K + 1):
        vals.append(abs(a_k(A, B, xi, A0, k)))
        vals.append(abs(a_k(B, A, xi, A0, k)))
    return max(vals)


def z_param(A, B, A0, gbar):
    """z = sqrt(2) alpha beta / (A_0 sqrt(gbar))  -- eq:z_guard"""
    return mp.sqrt(2) * A * B / (A0 * mp.sqrt(gbar))


def db(x):
    return mp.mpf(10)**(mp.mpf(x) / 10)
