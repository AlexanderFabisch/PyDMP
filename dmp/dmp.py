from numpy import *  # This is very ugly, but it is OK for our purposes :)

class DMP(object):
    def __init__(self, pastor_mod=False):
        self.pastor_mod = pastor_mod
        # Transformation system
        self.alpha = 25.0                # = D = 20.0
        self.beta = self.alpha / 4.0     # = K/D = 100.0/20.0 = 5.0
        # Canonical system
        self.alpha_t = self.alpha / 3.0
        # Obstacle avoidance
        self.gamma_o = 1000.0
        self.beta_o = 20.0 / pi

    def phase(self, tau, n_steps, t):
        """The phase variable replaces explicit timing. It starts with 1 at the
        beginning of the movement and converges exponentially to 0.
        """
        return exp(-self.alpha_t * linspace(0, tau, n_steps)[t] / tau)

    def spring_damper(self, x0, g, tau, s, X, Xd):
        """The transformation system generates a goal-directed movement."""
        if self.pastor_mod:
            # allows smooth adaption to goals, in the original version also the
            # forcing term is multiplied by a constant alpha * beta which you can
            # of course omit since the weights will simply be scaled
            return self.alpha * (self.beta * (g - X) - tau * Xd - self.beta * (g-x0) * s) / tau**2
        else:
            return self.alpha * (self.beta * (g - X) - tau * Xd) / tau**2

    def forcing_term(self, x0, g, tau, w, s, X, scale=False):
        """The forcing term shapes the movement arbitrarily based on the
        weights.
        """
        n_features = w.shape[1]
        c = exp(-self.alpha_t * linspace(0, tau, n_features) / tau)
        h = diff(c)
        h = hstack((h, [h[-1]]))
        phi = exp(-h*(s-c)**2)
        f = (phi * w / phi.sum()).sum(axis=1) * s
        if scale:
            f *= g - x0
    
        if X.ndim == 3:
            F = zeros_like(X)
            F[:, :] = f
            return F
        else:
            return f

    def obstacle(self, o, X, Xd):
        """Obstacle avoidance is based on point obstacles."""
        if X.ndim == 1:
          X = X[newaxis, newaxis, :]
          Xd = Xd[newaxis, newaxis, :]
        C = zeros_like(X)
        R = array([[cos(pi/2.0), -sin(pi/2.0)],
                   [sin(pi/2.0),  cos(pi/2.0)]])
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                theta = arccos((o-X[i, j]).dot(Xd[i, j]) / (linalg.norm(o-X[i, j]) * linalg.norm(Xd[i, j]) + 1e-10))
                C[i, j] = self.gamma_o * R.dot(Xd[i, j]) * theta * exp(-self.beta_o * theta)
        return squeeze(C)
