from numpy import *
from scipy import integrate
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import sys


"""
2015 Robert Gantner, SAM, ETH Zurich.
Convention:
    N     : number of elements
    N+1   : number of total vertices
    N-1   : number of internal vertices
    2*N+1 : number of total degrees of freedom
    2*N-1 : number of unknowns
"""

element_stiffness_fct = lambda h: array([[1, -1], [-1, 1]]) / (h)
element_mass_fct = lambda h: array([[2, 1], [1, 2]]) * h / 6.0

# basis functions (on reference interval [-1,1])
w = [lambda t: (1.0 - t) / 2.0, lambda t: (1.0 + t) / 2.0]
# derivatives of basis functions (on reference interval [-1,1])
dw = [lambda t: -0.5, lambda t: 0.5]


def element_stiffness_quadrature(mesh, coeff, k):
    """
    Construct element stiffness matrix of element k using quadrature,
    with coefficient function given by coeff(x)
    mesh: data structure with mesh.nodes and mesh.elements
    coeff: callable
    k: index of the element
    """
    x0 = mesh.nodes[mesh.elements[k][0]]
    x1 = mesh.nodes[mesh.elements[k][1]]
    h = x1 - x0
    Aloc = zeros((2, 2))
    for i in range(2):
        for j in range(2):
            Aloc[i, j] = (
                integrate.quad(
                    lambda x: coeff(((x0 + x1) + x * h) / 2.0) * dw[i](x) * dw[j](x),
                    -1,
                    1,
                )[0]
                * 2.0
                / h
            )
    return Aloc


def element_stiffness_pwconst(mesh, coeff, k):
    """
    Construct element stiffness matrix of element k using quadrature,
    with coefficient function given by coeff(x)
    mesh: data structure with mesh.nodes and mesh.elements
    coeff: list/array containing one value per element of the mesh
    k: index of the element
    """
    x0 = mesh.nodes[mesh.elements[k][0]]
    x1 = mesh.nodes[mesh.elements[k][1]]
    h = x1 - x0
    return coeff[k] * element_stiffness_fct(h)


def element_stiffness_pwlinear_logcoeff(mesh, coeff, k):
    """
    Construct element stiffness matrix of element k using quadrature,
    with coefficient function given by coeff(x)
    mesh: data structure with mesh.nodes and mesh.elements
    coeff: list/array containing one value per element of the mesh
    k: index of the element
    """
    x0 = mesh.nodes[mesh.elements[k][0]]
    x1 = mesh.nodes[mesh.elements[k][1]]
    h = x1 - x0
    return (
        exp((coeff[k] + coeff[k + 1]) / 2.0)
        * sinh((-coeff[k] + coeff[k + 1]) / 2.0)
        * 2.0
        / (-coeff[k] + coeff[k + 1])
        * element_stiffness_fct(h)
    )


def element_stiffness_const(mesh, coeff, k):
    """
    Construct element stiffness matrix of element k
    with coefficient given by coeff
    mesh: data structure with mesh.nodes and mesh.elements
    coeff: scalar value
    k: index of the element
    """
    x0 = mesh.nodes[mesh.elements[k][0]]
    x1 = mesh.nodes[mesh.elements[k][1]]
    h = x1 - x0
    return coeff * element_stiffness_fct(h)


def assemble_stiffness(mesh, coeff):
    ## Decide which element assembly routine to use
    ## They should accept a mesh, coefficient and element index k
    ## The coefficient will be of the form decided on below
    if hasattr(coeff, "__call__"):
        assemblytype = "quadrature"
        element_stiffness = element_stiffness_quadrature
    ##elif hasattr(coeff, '__getitem__'):
    ##    assemblytype = "pwconst"
    ##    element_stiffness = element_stiffness_pwconst
    elif hasattr(coeff, "__getitem__"):
        assemblytype = "pwlinear_logcoeff"
        element_stiffness = element_stiffness_pwlinear_logcoeff
    elif isinstance(coeff, (int, float, double)):
        assemblytype = "const"
        element_stiffness = element_stiffness_const
    else:
        raise Exception("Invalid coefficient type passed to 'assemble_stiffness'")

    N = len(mesh.elements)
    dofs = N + 1

    I = zeros(4 * N)
    J = zeros(4 * N)
    V = zeros(4 * N)
    ind = 0
    for k in range(N):
        Aloc = element_stiffness(mesh, coeff, k)
        for i in range(2):
            for j in range(2):
                I[ind] = k + i
                J[ind] = k + j
                V[ind] = Aloc[i, j]
                ind += 1
    A = sparse.csr_matrix((V, (I, J)), shape=(dofs, dofs))
    # csr has fast row slicing
    # csc has fast column slicing
    # conversion is fast
    # A = A[1:-1,:].tocsc()[:,1:-1]
    return A


def assemble_stiffness_periodicBC(mesh, coeff):
    ## Decide which element assembly routine to use
    ## They should accept a mesh, coefficient and element index k
    ## The coefficient will be of the form decided on below
    if hasattr(coeff, "__call__"):
        assemblytype = "quadrature"
        element_stiffness = element_stiffness_quadrature
    ##elif hasattr(coeff, '__getitem__'):
    ##    assemblytype = "pwconst"
    ##    element_stiffness = element_stiffness_pwconst
    elif hasattr(coeff, "__getitem__"):
        assemblytype = "pwlinear_logcoeff"
        element_stiffness = element_stiffness_pwlinear_logcoeff
    elif isinstance(coeff, (int, float, double)):
        assemblytype = "const"
        element_stiffness = element_stiffness_const
    else:
        raise Exception("Invalid coefficient type passed to 'assemble_stiffness'")

    N = len(mesh.elements)
    dofs = N

    I = zeros(4 * N)
    J = zeros(4 * N)
    V = zeros(4 * N)
    ind = 0
    for k in range(N):
        Aloc = element_stiffness(mesh, coeff, k)
        for i in range(2):
            for j in range(2):
                I[ind] = mod(k + i, N)
                J[ind] = mod(k + j, N)
                V[ind] = Aloc[i, j]
                ind += 1
    A = sparse.csr_matrix((V, (I, J)), shape=(dofs, dofs))
    # csr has fast row slicing
    # csc has fast column slicing
    # conversion is fast
    # A = A[1:-1,:].tocsc()[:,1:-1]
    return A


def element_mass_quadrature(mesh, coeff, k):
    """
    Construct element mass matrix of element k using quadrature,
    with coefficient function given by coeff(x)
    mesh: data structure with mesh.nodes and mesh.elements
    coeff: callable
    k: index of the element
    """
    x0 = mesh.nodes[mesh.elements[k][0]]
    x1 = mesh.nodes[mesh.elements[k][1]]
    h = x1 - x0
    Mloc = zeros((2, 2))
    for i in range(2):
        for j in range(2):
            Mloc[i, j] = (
                integrate.quad(
                    lambda x: coeff(((x0 + x1) + x * h) / 2.0) * w[i](x) * w[j](x),
                    -1,
                    1,
                )[0]
                * h
                / 2.0
            )
    return Mloc


def element_mass_pwconst(mesh, coeff, k):
    """
    Construct element mass matrix of element k using quadrature,
    with coefficient function given by coeff(x)
    mesh: data structure with mesh.nodes and mesh.elements
    coeff: list/array containing one value per element of the mesh
    k: index of the element
    """
    x0 = mesh.nodes[mesh.elements[k][0]]
    x1 = mesh.nodes[mesh.elements[k][1]]
    h = x1 - x0
    return coeff[k] * element_mass_fct(h)


def element_mass_const(mesh, coeff, k):
    """
    Construct element mass matrix of element k
    with coefficient given by coeff
    mesh: data structure with mesh.nodes and mesh.elements
    coeff: scalar value
    k: index of the element
    """
    x0 = mesh.nodes[mesh.elements[k][0]]
    x1 = mesh.nodes[mesh.elements[k][1]]
    h = x1 - x0
    return coeff * element_mass_fct(h)


def assemble_mass_periodicBC(mesh, coeff):
    ## Decide which element assembly routine to use
    ## They should accept a mesh, coefficient and element index k
    ## The coefficient will be of the form decided on below
    if hasattr(coeff, "__call__"):
        assemblytype = "quadrature"
        element_mass = element_mass_quadrature
    ##elif hasattr(coeff, '__getitem__'):
    ##    assemblytype = "pwconst"
    ##    element_mass = element_mass_pwconst
    elif isinstance(coeff, (int, float, double)):
        assemblytype = "const"
        element_mass = element_mass_const
    else:
        raise Exception("Invalid coefficient type passed to 'assemble_stiffness'")

    N = len(mesh.elements)
    dofs = N

    I = zeros(4 * N)
    J = zeros(4 * N)
    V = zeros(4 * N)
    ind = 0
    for k in range(N):
        Mloc = element_mass(mesh, coeff, k)
        for i in range(2):
            for j in range(2):
                I[ind] = mod(k + i, N)
                J[ind] = mod(k + j, N)
                V[ind] = Mloc[i, j]
                ind += 1
    M = sparse.csr_matrix((V, (I, J)), shape=(dofs, dofs))
    # csr has fast row slicing
    # csc has fast column slicing
    # conversion is fast
    return M


def assemble_rhs(mesh, fct=None):
    ## constant rhs function f(x)=1
    N = len(mesh.elements)
    dofs = N + 1
    # b_1 = array([1./3,4./3,1./3])*(h/2) # local vector
    if fct == None:
        return ones(dofs) * diff(mesh.nodes) / 2.0
    B = zeros(dofs)
    for k in range(N):
        x0 = mesh.nodes[mesh.elements[k][0]]
        x1 = mesh.nodes[mesh.elements[k][1]]
        h = x1 - x0
        for j in r_[:2]:
            B[k + j] += (
                integrate.quad(
                    lambda x: fct(((x0 + x1) + x * h) / 2.0) * w[j](x), -1, 1
                )[0]
                * h
                / 2
            )
    return B


def assemble_rhs_periodicBC(mesh, fct=None):
    ## constant rhs function f(x)=1
    N = len(mesh.elements)
    dofs = N
    # b_1 = array([1./3,4./3,1./3])*(h/2) # local vector
    if fct == None:
        return ones(dofs) * diff(mesh.nodes) / 2.0
    B = zeros(dofs)
    for k in range(N):
        x0 = mesh.nodes[mesh.elements[k][0]]
        x1 = mesh.nodes[mesh.elements[k][1]]
        h = x1 - x0
        for j in r_[:2]:
            B[mod(k + j, N)] += (
                integrate.quad(
                    lambda x: fct(((x0 + x1) + x * h) / 2.0) * w[j](x), -1, 1
                )[0]
                * h
                / 2
            )
    return B


def assemble_rhs_onept(mesh, fct):
    ## constant rhs function f(x)=1
    N = len(mesh.elements)
    dofs = N + 1
    # b_1 = array([1./3,4./3,1./3])*(h/2) # local vector
    B = zeros(dofs)
    for k in range(N):
        x0 = mesh.nodes[mesh.elements[k][0]]
        x1 = mesh.nodes[mesh.elements[k][1]]
        h = x1 - x0
        B[k] += fct(x0) * h
        B[k + 1] += fct(x1) * h
    return B


# def L2_projection(N,fct,M=None):
#    """
#    N: number of elements
#    fct: rhs function
#    M: mass matrix (if already constructed)
#    """
#    h = 1./N
#    # set up RHS vector
#    b = zeros(2*N+1)
#    for k in range(N):
#        for j in r_[:3]:
#            b[2*k+j] += integrate.quad(lambda x: fct(x)*w[j]((x-h*(k+0.5))*2./h), k*h, (k+1)*h)[0]
#            #b[2*k+j] += h*integrate.quad(lambda x: fct(h*(k+x))*w[j](2*x-1),0,1)[0]
#    # solve LSE
#    if M is None: M = assemble(element_mass_fct,N)
#    return solve(M,b)


def eval_fct(c, mesh, x):
    """
    Evalutate function at given points.
    c: coefficient vector of size 2*N+1
    mesh: mesh containing elements and nodes
    x: evaluation points
    """
    dofs = len(c)
    N = (dofs - 1) / 2
    assert N == len(mesh.elements)
    # initialize output
    y = zeros_like(x)
    i = 0  # index in x
    # loop over elements
    for k in range(N):
        x0 = mesh.nodes[mesh.elements[k][0]]
        x1 = mesh.nodes[mesh.elements[k][1]]
        stop = False
        i0 = i
        # count up to find last point still in the element
        for xx in x[i0:]:
            if xx <= x1:
                i += 1
            else:
                break
        if i == i0:
            continue  # nothing to do, no pts in element
        i1 = i
        xvals = x[i0:i1]  # list of points in this element
        for xx in xvals:
            assert xx <= x1
            assert xx >= x0
        # add contribution of each basis function
        X = (2 * xvals - (x0 + x1)) / (x1 - x0)  # map to reference element
        for j in r_[:3]:
            y[i0:i1] += w[j](X) * c[2 * k + j]
        if stop:
            break
    return y


def eval_fct_pt(u, mesh, x):
    """Evaluate the finite element function specified by coefficients u at the point x"""
    dofs = len(u)
    N = dofs - 1
    assert N == len(mesh.elements)
    # get index of element
    k = N - 1
    while (x < mesh.nodes[mesh.elements[k][0]]) and (k > 0):
        k -= 1
    x0 = mesh.nodes[mesh.elements[k][0]]
    x1 = mesh.nodes[mesh.elements[k][1]]
    assert x >= x0
    assert x < x1
    return u[k] + (u[k + 1] - u[k]) / (x1 - x0) * (x - x0)


def plot_function(c, mesh, x, **args):
    """c is coefficient vector of size 2*N+1; x are eval points (assume sorted)"""
    dofs = len(c)
    N = (dofs - 1) / 2
    assert N == len(mesh.elements)
    # initialize output
    y = eval_fct(c, mesh, x)
    plt.plot(x, y, **args)


def plot_const(coeff, mesh, **args):
    """plot piecewise constant function, given by coefficients in 'coeff'"""
    x = zeros(2 * len(mesh.nodes) - 2)
    y = zeros_like(x)
    x[0] = mesh.nodes[0]
    y[0] = coeff[0]
    x[-1] = mesh.nodes[-1]
    y[-1] = coeff[-1]
    for i in range(1, len(mesh.nodes) - 1):
        x[2 * i - 1] = mesh.nodes[i]
        x[2 * i] = mesh.nodes[i]
        y[2 * i - 1] = coeff[i - 1]
        y[2 * i] = coeff[i]
    plt.plot(x, y, **args)


def L2error(u, mesh, exact):
    dofs = len(u)
    N = dofs - 1
    assert N == len(mesh.elements)
    err = 0.0
    # for k in range(N):
    #    x0 = mesh.nodes[mesh.elements[k][0]]
    #    x1 = mesh.nodes[mesh.elements[k][1]]
    err = integrate.quad(
        lambda x: (eval_fct_pt(u, mesh, x) - exact(x)) ** 2.0, mesh.D[0], mesh.D[1]
    )[0]
    err = sqrt(err)
    return err


def pwconst(c, mesh, x):
    assert len(c) == len(mesh.elements)
    if isinstance(x, (int, float, double)):
        x = array([x])
    y = zeros_like(x)
    for j in range(len(x)):
        if x[j] == mesh.D[1]:
            y[j] = c[-1]
        else:
            i = 0
            for k in range(len(mesh.elements)):
                if (
                    x[j] < mesh.nodes[mesh.elements[k][0]]
                    or x[j] >= mesh.nodes[mesh.elements[k][1]]
                ):
                    continue
                else:
                    i = k
                    break
            y[j] = c[i]
    return y


def solve_pde(N, coeffs, f, mesh):
    """
    Solve the parametric diffusion equation for coefficient 'coeff'
    N: number of elements
    coeffs: coefficient. either a function of x, or a vector of coefficients (if pw const)
    f: right-hand side
    assemblytype: one of: ["quadrature", "pwconst", "pwconstsparse", "const"]
    mesh: one of ["equidistant", "geometric"]
    """
    ## assembly
    # 1. matrices
    A = assemble_stiffness(mesh, coeffs)
    A = A[1:-1, :].tocsc()[:, 1:-1]  # remove first and last rows and columns
    # print A.toarray()

    ## assemble right-hand side with one-point quadrature
    ##F = assemble_rhs_onept(mesh,f)
    F = assemble_rhs(mesh, f)

    ## L2 projection of f onto L^2(D)
    # M = assemble(element_mass_fct,N)
    # f_coeff = solve(M,f)

    ## Solve the linear system
    # initialize with boundary conditions
    u = zeros(N + 1)
    # solve for interior degrees of freedom
    # u[1:-1] = solve(abar*A[1:-1,1:-1],f[1:-1])
    u[1:-1] = spsolve(A, F[1:-1])

    del A
    del F

    return u


def gen_mesh(D, N, meshtype):
    """Generates a mesh of the interval D=(a,b) with N elements"""

    class Mesh:
        def __init__(self, nodes, elements, D=(0, 1)):
            self.nodes = array(nodes)
            self.elements = array(elements)
            self.D = D

    if meshtype == "equidistant":
        nodes = linspace(D[0], D[1], N + 1)
        elements = [(i, i + 1) for i in r_[:N]]
    elif meshtype == "geometric":
        nodes = [D[0] + (D[1] - D[0]) * (2**k - 1) / 2.0**k for k in r_[:N]]
        nodes.append(D[1])
        elements = [(i, i + 1) for i in r_[:N]]
    elif (
        meshtype == "equidistant2"
    ):  # first element is [0,0.5], rest split into N pieces
        nodes = hstack([D[0], linspace((D[0] + D[1]) / 2.0, D[1], N)])
        elements = [(i, i + 1) for i in r_[:N]]
    elif meshtype == "log":
        nodes = 1 - (logspace(0, 1, N + 1)[::-1] - 1) / 9
        elements = [(i, i + 1) for i in r_[:N]]
    elif meshtype == "sqrt":
        nodes = sqrt(linspace(0, 1, N + 1))
        elements = [(i, i + 1) for i in r_[:N]]
    elif meshtype == "0.2":  # graded mesh
        nodes = linspace(0, 1, N + 1) ** 0.2
        elements = [(i, i + 1) for i in r_[:N]]
    else:
        raise Exception("Unsupported mesh type: " + meshtype)
    mesh = Mesh(nodes, elements, D)
    return mesh


def plot_mesh(mesh, **args):
    y = zeros_like(mesh.nodes)
    plt.plot(mesh.nodes, y, "o", **args)


if __name__ == "__main__":

    Nce = 4
    Nc = 2**Nce + 1
    abar = 1.0
    theta = 0.85
    # coeffs = abar + (2*random.random(Nc)-1) / r_[1:Nc+1]**1.
    # coeffs = abar + 0.9*ones(Nc) / r_[1:Nc+1]**2.
    coeffs = 1 - theta / r_[1 : Nc + 1] ** 2.0

    # right-hand side
    c = 15.0
    n = 0.0
    fct = lambda x: c * x**n
    exact = lambda x: c * (x - x ** (n + 2.0)) / ((n + 1.0) * (n + 2))

    # fct = lambda x: pi*pi*sin(pi*x)
    # exact = lambda x: sin(pi*x)
    # exact solution is scaled with diffusion coefficient (if constant and not 1)
    # exact_fct = lambda x: exact(x)/abar

    # for plotting
    xvals = linspace(0, 1, 1001)

    do_cppcompare = False
    if do_cppcompare:
        N = 2**3
        # y = 2*random.random(N)-1
        # y = array([-0.509766, 0.585938, 0.109375, 0.769531, -0.310547, 0.197266, -0.175781, 0.197266])
        y = -ones(N)
        coeffs = 1.0 + 0.85 * y * r_[1 : N + 1] ** -2.0
        print("coeffs:", coeffs)
        meshtype = "0.2"
        mesh = gen_mesh((0, 1), N, meshtype)
        # f = assemble_rhs_onept(mesh,fct)
        # print "f:",f
        # A = assemble_stiffness(mesh, coeffs)
        # print "A:",A.toarray()
        u = solve_pde(N, coeffs, fct, mesh)
        print(eval_fct(u, mesh, array([0.2, 0.5, 0.7])))
        plot_function(u, mesh, xvals, label="this")
        u2 = array(
            [
                0,
                6.728451305369703,
                2.575020569537857,
                2.280359038729515,
                1.939866750080319,
                1.725156250347434,
                1.493474440358523,
                1.314144471945334,
                1.125431448442045,
                0.9676230668894894,
                0.8036865986557135,
                0.6609122202894996,
                0.5137486876239928,
                0.3823142818570656,
                0.2475453801037013,
                0.1250923500794603,
                0,
            ]
        )
        plot_function(u2, mesh, xvals, label="other")
        plt.legend(loc="best")
        plt.show()
        # print "u:",u

    do_test = False
    if do_test:
        # meshtype = "geometric"
        # meshtype = "equidistant"
        # meshtype = "equidistant2"
        meshtype = "log"
        mesh = gen_mesh((0, 1), Nc, meshtype)
        # plot coeff
        # plot_const(coeffs,mesh)
        # plt.show()
        # solve
        ## 1
        u = solve_pde(Nc, coeffs, fct, mesh)
        plot_function(u, mesh, xvals, label="N=%d" % Nc)
        plot_mesh(mesh, markersize=10)
        plot_const(coeffs, mesh, color="0.2", label="coeff")
        ## 2
        N2 = 2 * (Nc - 1) + 1
        mesh2 = gen_mesh((0, 1), N2, meshtype)
        u = solve_pde(N2, lambda x: pwconst(coeffs, mesh, x), fct, mesh2)
        plot_function(u, mesh2, xvals, label="N=%d" % N2)
        plot_mesh(mesh2, markersize=8)
        ### 3
        # N3 = 2*(N2-1)+1
        # mesh3 = gen_mesh((0,1),N3,meshtype)
        # u = solve_pde(N3, lambda x: pwconst(coeffs,mesh,x), fct, mesh3)
        # plot_function(u,mesh3,xvals,label="N=%d"%N3)
        # plot_mesh(mesh3,markersize=6)
        ## exact
        plt.plot(xvals, exact(xvals), "--k", linewidth=2, label="exact")
        ## done
        plt.legend(loc="best")
        plt.show()

    do_conv = False
    if do_conv:
        meshtype = "equidistant2"
        # Exact solution. Since mesh is nested, can use pw const assembly.
        # Need to construct coefficient vector.
        mesh_coeff = gen_mesh((0, 1), Nc, meshtype)
        Nex = 2 ** (Nce + 3) + 1
        coeffs_exact = zeros(Nex)
        coeffs_exact[0] = coeffs[0]
        Nblock = (Nex - 1) / (Nc - 1)
        for k in range(Nc - 1):
            coeffs_exact[1 + k * Nblock : 1 + (k + 1) * Nblock] = coeffs[1 + k]
        mesh_exact = gen_mesh((0, 1), Nex, meshtype)
        exact = solve_pde(Nex, coeffs_exact, fct, mesh_exact)
        xvals = linspace(0, 1, 1001)
        plot_function(exact, mesh_exact, xvals, label="exact")
        plot_const(coeffs_exact, mesh_exact, ls="-", color="0.2", label="coeff exact")

        # compute L2 error convergence
        e_L2 = []
        for Ne in r_[1:Nce]:
            # these are larger meshes than the one on which the coefficient is located.
            N = 2**Ne + 1
            mesh = gen_mesh((0, 1), N, meshtype)
            u = solve_pde(N, lambda x: pwconst(coeffs, mesh_coeff, x), fct, mesh)
            ##
            xvals = linspace(0, 1, 1001)
            # plt.plot(xvals,pwconst(coeffs,mesh_coeff,xvals))
            plot_function(u, mesh, xvals, label="$N=%d$" % N)
            plot_mesh(mesh)
            ##
            e = L2error(u, mesh, lambda x: eval_fct_pt(exact, mesh_exact, x))
            e_L2.append(e)
        for Ne in r_[Nce : Nce + 3]:
            # starting here, the solution can be computed exactly
            N = 2**Ne + 1
            mesh = gen_mesh((0, 1), N, meshtype)
            coeffs_new = zeros(N)
            Nblock = (N - 1) / (Nc - 1)
            coeffs_new[0] = coeffs[0]
            for k in range(Nc - 1):
                coeffs_new[1 + k * Nblock : 1 + (k + 1) * Nblock] = coeffs[1 + k]
            u = solve_pde(N, lambda x: pwconst(coeffs, mesh_coeff, x), fct, mesh)
            # u = solve_pde(N, coeffs_new, fct, mesh)
            ##
            # plot_const(coeffs_new,mesh)
            plot_function(u, mesh, xvals, label="$N=%d$" % N)
            plot_mesh(mesh)
            ##
            e = L2error(u, mesh, lambda x: eval_fct_pt(exact, mesh_exact, x))
            e_L2.append(e)
        print(e_L2)
        plt.legend(loc="best")
        # plt.show()
        plt.figure()
        plt.loglog(2.0 ** r_[1 : Nce + 3], e_L2, "-o")
        plt.grid(True)
        plt.show()

    do_range = True
    if do_range:
        # visualize range
        plt.figure()
        meas_pts = array([0.2, 0.5, 0.7])
        y = {}
        if len(sys.argv) > 1:
            s = int(sys.argv[1])
        else:
            s = 8
        print("N=s=%d" % s)
        N = s
        meshtype = "0.2"
        # meshtype = "equidistant2"
        mesh = gen_mesh((0, 1), N, meshtype)
        xvals = hstack([linspace(0, 0.499, 100), linspace(0.5, 1, 2 * s + 1)])
        for t in [-1, 1]:
            coeffs = abar + t * theta * ones(N) / r_[1 : N + 1] ** 2.0
            # coeffs = hstack([coeffs_old[::2], coeffs_old[-1::-2]])
            # coeffs = hstack([coeffs_old[0], coeffs_old[-1:0:-1]])
            u = solve_pde(N, coeffs, fct, mesh)
            # u = solve_pde(N, coeffs, fct, "pwconst")
            y[t] = eval_fct(u, mesh, meas_pts)
            print("solved for t=", t)
            lab = {-1: r"$q(u_-,\cdot)$", 1: r"$q(u_+,\cdot)$"}[t]
            plot_function(u, mesh, xvals, label=lab)
            lab = {-1: "$u_-$", 1: "$u_+$"}[t]
            col = {-1: "0.", 1: "0.5"}[t]
            plot_const(coeffs, mesh, color=col, label=lab)
        plt.plot(vstack([meas_pts, meas_pts]), vstack([y[1], y[-1]]), "r-", linewidth=2)
        M = (y[1] + y[-1]) / 2
        diff = y[-1] - y[1]
        print("range:", zip(y[1], y[-1]))
        D = mean(diff)
        print("diffs:", diff, "mean[diff]:", D)
        gamma1 = D**2 / 4  # 1\sigma
        gamma2 = D**2 / 16  # 2\sigma
        print("Gamma_1= %1.8f" % gamma1)
        print("Gamma_2= %1.8f" % gamma2)
        for i in range(len(meas_pts)):
            plt.text(meas_pts[i] + 0.01, M[i] - 0.03, "%1.2f" % diff[i])

        # misc
        a = plt.axis()
        plt.axis((a[0], a[1], 0, a[3]))
        plt.xlabel("Spatial Domain $D=(0,1)$", fontsize=16)
        plt.ylabel("Solution and Coefficients", fontsize=16)
        plt.title(r"Solution to $-(a(x)u'(x))' = f(x)$ with $s=N=%d$" % N, fontsize=16)
        plt.grid(True)
        plt.legend(loc="best")
        figfile = "images/uplusminus_%s_s%d" % (meshtype, s)
        figfile = figfile.replace(".", "") + ".pdf"
        print("saving to:", figfile)
        plt.savefig(figfile)
        plt.show()

    # 5. plots
    do_plot = False
    if do_plot:
        xvals = linspace(0, 1, 1001)
        # plot_function(f_coeff,xvals)
        ## coeffs
        plt.plot(xvals, pwconst(coeffs, mesh, xvals))
        plot_function(u, xvals, linewidth=2)
        plt.plot(xvals, exact_fct(xvals), "--k")
        # legend, etc
        # plt.legend(["$f(x)$", "$a(x)$", "$u_h(x)$", "$u(x)$"],loc="best")
        plt.legend(["$a(x)$", "$u_h(x)$"], loc="best")
        plt.legend(["$a(x)$", "$u_h(x)$", "$u(x)$"], loc="best")
        # plt.legend(["$f(x)$", "$f_2(x)$", "$u_h(x)$", "$u(x)$"],loc="best")
        plt.title(r"Solution to $-\frac{d^2u}{dx^2}(x) = f(x)$")
        plt.grid(True)
        # plt.axis((0,1,0,1))
        plt.show()
