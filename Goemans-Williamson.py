#код's not stolenbut rather borrowed
'''maxcut.py: A simple implementation using CVXPY of Goemans-Williamson
algorithm for Max-Cut problem of finding a maximum cut of a graph:
  Goemans and Williamson. Improved approximation algorithms for maximum cut and
  satisfiability problems using semidefinite programming. J. ACM, 42(6):1115–1145, 1995.
  http://www-math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf

Algorithm is randomized and returns a random subset S, but with guarantee on S:
    cut-size(S) > 0.878 * max-cut-size

Example. For a cycle on 5 vertices [0, 1, 2, 3, 4] we get:
  Chosen subset: [1 4]
  Cut size: 4
  Edges of the cut: [(0, 1), (1, 2), (3, 4), (0, 4)]
'''

import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm

def gw(n, edges):
    '''Goemans-Williamson algorithm for Max-Cut:
    Given a graph G(V=[n], E=edges), returns a vector x \in {-1, 1}^n
    that corresponds to the chosen subset of vertices S of V. '''
    ## SDP Relaxation
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [
        X[i, i] == 1 for i in range(n)
    ]
    objective = sum( 0.5*(1 - X[i, j]) for (i, j) in edges )
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()

    ## Hyperplane Rounding
    Q = sqrtm(X.value).real
    r = np.random.randn(n)
    x = np.sign(Q @ r)

    return x

def cut(x, edges):
    '''Given a vector x \in {-1, 1}^n and edges of a graph G(V=[n], E=edges),
    returns the edges in cut(S) for the subset of vertices S of V represented by x.'''
    xcut = []
    for i, j in edges:
        if np.sign(x[i]*x[j]) < 0:
            xcut.append((i, j))
    return xcut


def parse_graph_file(filename):
    vertex_names = []
    vertex_to_index = {}
    edges = []

    with open(filename, 'r') as f:
        f.readline()
        line = f.readline().strip()
        while line and line != 'E':
            name = line.strip().strip('()')
            print(name)
            vertex_names.append(name)
            line = f.readline().strip()

        for i, name in enumerate(vertex_names):
            vertex_to_index[name] = i

        for line in f:
            line = line.strip()
            if not line:
                continue
            #   (uN)-(vN)-cost
            parts = line.split('-')
            uN = parts[0].strip('()')
            vN = parts[1].strip('()')
            cost = int(parts[2])
            u_index = vertex_to_index[uN]
            v_index = vertex_to_index[vN]
            edges.append((u_index, v_index))

    return vertex_names, edges

vertex_names, edges = parse_graph_file('input')
n = len(vertex_names)
print("Vertices:", vertex_names)
print("Edges:", edges)


x = gw(n, edges)

xcut = cut(x, edges)

print('Cut subgraphs: %s and %s' % (np.where(x == 1)[0], np.where(x == -1)[0]))
print('Max-cut: %i' % len(xcut) )
print('Cut edges: %s' % xcut )

