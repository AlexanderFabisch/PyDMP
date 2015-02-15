import numpy as np


class DMP(object):
    def __init__(self, pastor_mod=False):
        self.pastor_mod = pastor_mod
        # Transformation system
        self.alpha = 25.0             # = D = 20.0
        self.beta = self.alpha / 4.0  # = K / D = 100.0 / 20.0 = 5.0
        # Canonical system
        self.alpha_t = self.alpha / 3.0
        # Obstacle avoidance
        self.gamma_o = 1000.0
        self.beta_o = 20.0 / np.pi

    def phase(self, tau, n_steps, t):
        """The phase variable replaces explicit timing.

        It starts with 1 at the beginning of the movement and converges
        exponentially to 0.
        """
        return np.exp(-self.alpha_t * np.linspace(0, tau, n_steps)[t] / tau)

    def spring_damper(self, x0, g, tau, s, X, Xd):
        """The transformation system generates a goal-directed movement."""
        if self.pastor_mod:
            # Allows smooth adaption to goals, in the original version also the
            # forcing term is multiplied by a constant alpha * beta which you
            # can of course omit since the weights will simply be scaled
            mod = -self.beta * (g - x0) * s
        else:
            mod = 0.0
        return self.alpha * (self.beta * (g - X) - tau * Xd + mod) / tau ** 2

    def forcing_term(self, x0, g, tau, w, s, X, scale=False):
        """The forcing term shapes the movement based on the weights."""
        n_features = w.shape[1]
        h = np.exp(-self.alpha_t * np.linspace(0, tau, n_features) / tau)
        c = np.diff(h)
        c = np.hstack((c, [c[-1]]))
        phi = np.exp(-h * (s - c) ** 2)
        f = (phi * w / phi.sum()).sum(axis=1) * s
        if scale:
            f *= g - x0
    
        if X.ndim == 3:
            F = np.empty_like(X)
            F[:, :] = f
            return F
        else:
            return f

    def obstacle(self, o, X, Xd):
        """Obstacle avoidance is based on point obstacles."""
        if X.ndim == 1:
          X = X[np.newaxis, np.newaxis, :]
        if Xd.ndim == 1:
          Xd = Xd[np.newaxis, np.newaxis, :]

        C = np.zeros_like(X)
        R = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                      [np.sin(np.pi / 2.0),  np.cos(np.pi / 2.0)]])
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                obstacle_diff = o - X[i, j]
                theta = (np.arccos(obstacle_diff.dot(Xd[i, j]) /
                                   (np.linalg.norm(obstacle_diff) *
                                    np.linalg.norm(Xd[i, j]) + 1e-10)))
                C[i, j] = (self.gamma_o * R.dot(Xd[i, j]) * theta *
                           np.exp(-self.beta_o * theta))

        return np.squeeze(C)


def trajectory(dmp, w, x0, g, dt, tau, n_steps, o, shape, avoidance, verbose=0):
    if verbose >= 1:
        print("Trajectory with x0 = %s, g = %s, tau=%.2f, dt=%.3f, n_steps=%d"
              % (x0, g, tau, dt, n_steps))
    x = x0.copy()
    xd = np.zeros_like(x, dtype=np.float64)

    X = [x0.copy()]
    Xd = [xd.copy()]
    for t in xrange(n_steps):
        s = dmp.phase(tau, n_steps, t)
        sd = dmp.spring_damper(x0, g, tau, s, x, xd)
        f = dmp.forcing_term(x0, g, tau, w, s, x) if shape else 0.0
        C = dmp.obstacle(o, x, xd) if avoidance else 0.0
        xd += dt * (sd + f + C)
        x += dt * xd
        X.append(x.copy())
        Xd.append(xd.copy())
    return np.array(X), np.array(Xd)


def potential_field(dmp, t, v, w, x0, g, tau, n_steps, o, x_range, y_range,
                    n_tics):
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], n_tics),
                         np.linspace(y_range[0], y_range[1], n_tics))
    x = np.array((xx, yy)).transpose((1, 2, 0))
    xd = np.empty_like(x)
    xd[:, :] = v

    s = dmp.phase(tau, n_steps, t)
    sd = dmp.spring_damper(x0, g, tau, s, x, xd)
    f = dmp.forcing_term(x0, g, tau, w, s, x)
    C = dmp.obstacle(o, x, xd)
    acc = sd + f + C
    return xx, yy, sd, f, C, acc
