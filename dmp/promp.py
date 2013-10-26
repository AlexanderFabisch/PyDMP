from numpy import *  # This is very ugly, but it is OK for our purposes :)

class ProMP(object):
    def __init__(self):
        pass

    def phase(self, tau, n_steps, t):
        """The phase variable replaces explicit timing. It starts with 0 at the
        beginning of the movement ends at 1.
        """
        return linspace(0, tau, n_steps)[t]

    def step(self, tau, w, s):
        n_features = w.shape[1]
        c = linspace(0, tau, n_features)
        h = diff(c)
        h = hstack((h, [h[-1]]))
        phi = exp(-h*(s-c)**2)
        phi_der = 2*h*(c-s)*phi / tau
        y = w.dot(phi)  # Usually we have to normalize the basis functions
        yd = w.dot(phi_der)
        return y, yd

    def sample(self, tau, w, Sigma, s):
        y, yd = self.step(tau, w, s)
        eps = random.multivariate_normal(zeros(y.shape[0] + yd.shape[0]), Sigma)
        return (y+eps[:y.shape[0]], yd+eps[y.shape[0]:])

    def imitate(self, demos):
        pass
        # TODO implement