from sympy import *


# Computes the symbolic expressions for fiber strain and derivative of 
# fiber strain as a function of fiber stress

A, B, sig = symbols('A B sig')

eps = A + (A*(sig - ((A - 1)*sqrt(B*(2*A - 1)))/(2*A -1)))/sqrt(B + (sig - ((A - 1)*sqrt(B*(2*A-1)))/(2*A - 1))**2) - 1
eps_prime = simplify(diff(eps, sig))

print(eps)
print(eps_prime)