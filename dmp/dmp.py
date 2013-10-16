from numpy import *  # This is very ugly, but it is OK for our purposes :)

class DMP(object):
    def __init__(self, pastor_mod=False):
        self.pastor_mod = pastor_mod

        self.alpha = 25.0 # = D = 20.0
        self.beta = self.alpha / 4.0 # = K/D = 100.0/20.0 = 5.0
        self.alpha_t = self.alpha / 3.0

        self.gamma_o = 1000.0
        self.beta_o = 20.0 / pi

    def phase(self, tau, n_steps, t):
        return exp(-self.alpha_t * linspace(0, tau, n_steps)[t] / tau)

    def spring_damper(self, x0, g, tau, s, X, Xd):
        # TODO Muelling's DMP
        if self.pastor_mod:
            # allows smooth adaption to goals, in the original version also the
            # forcing term is multiplied by a constant alpha * beta which you can
            # of course omit since the weights will simply be scaled
            return self.alpha * (self.beta * (g - X) - tau * Xd - self.beta * (g-x0) * s) / tau**2
        else:
            return self.alpha * (self.beta * (g - X) - tau * Xd) / tau**2

    def forcing_term(self, x0, g, tau, w, s, X, scale=False):
        n_features = w.shape[1]
        h = exp(-self.alpha_t * linspace(0, tau, n_features) / tau)
        c = diff(h)
        c = hstack((c, [c[-1]]))
        phi = exp(-h*(s-c)**2)
        f = (phi * w / phi.sum()).sum(axis=1) * s
        if scale:
            f *= g - x0
    
        if len(X.shape) == 3:
            F = zeros_like(X)
            F[:, :, 0] = f[0]
            F[:, :, 1] = f[1]
            return F
        else:
            return f

    def obstacle(self, o, X, Xd):
        R = array([[cos(pi/2.0), -sin(pi/2.0)],
                   [sin(pi/2.0),  cos(pi/2.0)]])
        if len(X.shape) == 3:
            C = zeros_like(X)
            for i in xrange(X.shape[0]):
                for j in xrange(X.shape[1]):
                    theta = arccos((o-X[i, j]).dot(Xd[i, j]) / (linalg.norm(o-X[i, j]) * linalg.norm(Xd[i, j]) + 1e-10))
                    C[i, j] = self.gamma_o * R.dot(Xd[i, j]) * theta * exp(-self.beta_o * theta)
            return C
        else:
            theta = arccos((o-X).dot(Xd) / (linalg.norm(o-X) * linalg.norm(Xd) + 1e-10))
            C = self.gamma_o * R.dot(Xd) * theta * exp(-self.beta_o * theta)
            return C
