import numpy as np
rng = np.random.default_rng(7)
N_av = 1_200_000
cap  = 5_000_000          # high cap; censored avalanches excluded from fits
MAXGEN = 500_000

sizes = np.empty(N_av, dtype=np.int64)
durs  = np.empty(N_av, dtype=np.int64)
cens  = np.zeros(N_av, dtype=bool)

for i in range(N_av):
    active = 1; S = 1; T = 1; g = 0
    while active > 0:
        active = rng.poisson(active)
        if active > 0:
            S += active; T += 1
        g += 1
        if S >= cap or g >= MAXGEN:
            cens[i] = True; break
    sizes[i] = S; durs[i] = T

ok = ~cens
S = sizes[ok]; Tn = durs[ok]
frac_cens = cens.mean()

def csn_mle(x, xmin):
    x = x[x >= xmin]; n = x.size
    return 1.0 + n/np.sum(np.log(x/(xmin-0.5))), n

def ks(x, xmin, tau):
    x = np.sort(x[x >= xmin]); n = x.size
    Femp = np.arange(1,n+1)/n
    Ffit = 1.0-(x/xmin)**(-(tau-1.0))
    return np.max(np.abs(Femp-Ffit))

tau, nt   = csn_mle(S, 8)
alpha, na = csn_mle(Tn, 8)
Dtau, Dal = ks(S,8,tau), ks(Tn,8,alpha)

# gamma from <S>(T) in an un-truncated window
Ts, mS = [], []
for t in range(5, 120):
    m = S[Tn == t]
    if m.size >= 100:
        Ts.append(t); mS.append(m.mean())
Ts, mS = np.array(Ts), np.array(mS)
A = np.vstack([np.log(Ts), np.ones_like(Ts,float)]).T
gamma = np.linalg.lstsq(A, np.log(mS), rcond=None)[0][0]
pred = (alpha-1)/(tau-1)

print(f"avalanches {N_av:,} | censored {frac_cens*100:.3f}% | maxS {sizes.max():,} maxT {durs.max():,}")
print(f"tail n: S>=8 {nt:,} | T>=8 {na:,}")
print("-"*50)
print(f"tau_hat   (size)     {tau:.4f}   target 1.5000   KS D={Dtau:.4f}")
print(f"alpha_hat (duration) {alpha:.4f}   target 2.0000   KS D={Dal:.4f}")
print(f"gamma_hat <S>(T)     {gamma:.4f}   target 2.0000")
print("-"*50)
print(f"(alpha-1)/(tau-1)    {pred:.4f}   vs gamma_hat {gamma:.4f}   gap {abs(gamma-pred):.4f}")
