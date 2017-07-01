from sympy import *

from sympy import Symbol, diff

D = Symbol('D', real=True, nonzero=True)
pi = Symbol('rho_i', real=True, nonzero=True)
h = Symbol('h', real=True, nonzero=True)
pw = Symbol('rho_w', real=True, nonzero=True)
g = Symbol('g', real=True, nonzero=True)
t = Symbol('t', real=True, nonzero=True)
q = Symbol('q', real=True, nonzero=True)
r = Symbol('r', real=True, nonzero=True)
mu = Symbol('mu')
H = Symbol('H', real=True, nonzero=True)
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
v = Symbol('v', real=True)
tf = Symbol('tau_phi', real=True, nonzero=True)
lm = Symbol('lambda')
eta = Symbol('eta')
Phile = Symbol('Phi_le')
xi = Symbol('xi', real=True, nonzero=True)
chi = Symbol('chi', real=True, nonzero=True)
a = Symbol('a', real=True, nonzero=True)
b = Symbol('b', real=True, nonzero=True)
A = Symbol('A', real=True, nonzero=True)
B = Symbol('B', real=True, nonzero=True)
P = Symbol('P', real=True, nonzero=True)
Fr = Symbol('F', real=True, nonzero=True)
delta = DiracDelta(x) * DiracDelta(y) * P * exp(I * Fr * t)

k = sqrt(lm ** 2 + eta ** 2)


def nabla4(func):
    d4_dx4 = diff(func, x, 4)
    d4_dy4 = diff(func, y, 4)
    d4_dx2_dy2 = diff(diff(func, x, 2), y, 2)
    return d4_dx4 + 2 * d4_dx2_dy2 + d4_dy4


delta = inverse_fourier_transform(delta, x, lm).doit()
delta = inverse_fourier_transform(delta, y, eta).doit()
delta *= exp(-I * (lm * x + eta * y))
pprint(delta)
w = Function('omega')(lm, eta, t) * delta
Phi_le = w / (k * sinh(k * H))
Phi = Phi_le * cosh((H + z) * k).subs(z, 0).doit()

model = D * (nabla4(w) + tf * diff(nabla4(w), t)) + pw * g * w + pi * h * diff(w, t, 2) + pw * diff(Phi,
                                                                                                    t) + delta
model = model.collect(P).simplify()
# model = solve(model, w)
if __name__ == "__main__":
    fn = Function('Fn')(x)
    equ = fn * 2 + x / fn - 3 * sin(x + 1)
    pprint(solve(equ, fn))
    # pprint(model)
