import autograd
import autograd.numpy as np
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

def hamiltonian_fn(coords, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g = 1):
    q1, q2, p1, p2 = np.split(coords, 4)
    
    # Inertia terms
    # denom = m1 + m2 * np.sin(q1 - q2) ** 2
    H = (
        (p1**2 * m2 * l2**2 + p2**2 * (m1 + m2) * l1**2 - 2 * m2 * l1 * l2 * p1 * p2 * np.cos(q1 - q2))/2 
        
    )
    H += -(m1 + m2) * g * l1 * np.cos(q1) - m2 * g * l2 * np.cos(q2)
    return H

def dynamics_fn(t,coords, m1=1.0, m2=1.0, l1=1.0, l2=1.0):
    #print(coords.shape)
    dH = autograd.grad(hamiltonian_fn)(coords)
    dq1dt, dq2dt, dp1dt, dp2dt = np.split(dH, 4)
    return np.concatenate([dp1dt, dp2dt, -dq1dt, -dq2dt])

def get_trajectory(t_span=[0, 10], timescale=100, y0=None,radius = None, noise_std=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))
    
    # Initial state
    if y0 is None:
        y0 = np.random.rand(4) * 2.0 - 1.0
    #print(y0)
    if radius is None:
        radius = np.random.rand() + 1.3 # sample a range of radii
    y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius
    
    ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    #print(ivp)
    q1, q2, p1, p2 = ivp["y"] 
    dydt = np.stack([dynamics_fn(None, y) for y in ivp['y'].T]).T
    dq1dt, dq2dt, dp1dt, dp2dt = np.split(dydt, 4)
    
    # Add noise
    q1 += np.random.randn(*q1.shape) * noise_std
    q2 += np.random.randn(*q2.shape) * noise_std
    p1 += np.random.randn(*p1.shape) * noise_std
    p2 += np.random.randn(*p2.shape) * noise_std
    
    return q1, q2, p1, p2, dq1dt, dq2dt, dp1dt, dp2dt, t_eval

def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
    np.random.seed(seed)
    xs, dxs = [], []
    
    for _ in range(samples):
        q1, q2, p1, p2, dq1dt, dq2dt, dp1dt, dp2dt, _ = get_trajectory(**kwargs)
        xs.append(np.stack([q1,q2, p1,p2]).T)
        dxs.append(np.stack([dq1dt,dq2dt, dp1dt,dp2dt]).T)
    
    data = {'x': np.concatenate(xs), 'dx': np.concatenate(dxs).squeeze()}

    
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data


def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten(), np.zeros_like(b.flatten()), np.zeros_like(a.flatten())])
    dydt = np.stack([dynamics_fn(None, y) for y in ys.T]).T
    return {'x': ys.T, 'dx': dydt.T}

# import autograd
# import autograd.numpy as np
# from autograd import grad
# import scipy.integrate


# solve_ivp = scipy.integrate.solve_ivp

# # --- HAMILTONIAN FORMULATION ---
# def hamiltonian_fn(coords, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=1.0):
#     q1, q2, p1, p2 = np.split(coords, 4)
#     H = (
#         (p1**2 * m2 * l2**2 + p2**2 * (m1 + m2) * l1**2 - 2 * m2 * l1 * l2 * p1 * p2 * np.cos(q1 - q2)) / 2
#     )
#     H += -(m1 + m2) * g * l1 * np.cos(q1) - m2 * g * l2 * np.cos(q2)
#     return H

# def hamiltonian_dynamics(t, coords):
#     dH = autograd.grad(hamiltonian_fn)(coords)
#     dq1dt, dq2dt, dp1dt, dp2dt = np.split(dH, 4)
#     return np.concatenate([dp1dt, dp2dt, -dq1dt, -dq2dt])

# # --- LAGRANGIAN FORMULATION ---
# def lagrangian_fn(q, dq, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=1.0):
#     q1, q2 = q
#     dq1, dq2 = dq

#     T = 0.5 * m1 * (l1 * dq1)**2 + 0.5 * m2 * ((l1 * dq1)**2 + (l2 * dq2)**2 + 2 * l1 * l2 * dq1 * dq2 * np.cos(q1 - q2))
#     y1 = -l1 * np.cos(q1)
#     y2 = y1 - l2 * np.cos(q2)
#     V = m1 * g * y1 + m2 * g * y2
#     return T - V



# def lagrangian_dynamics(t, state):
#     q = state[:2]
#     dq = state[2:]

#     # Replace np operations with autograd.numpy (anp)
#     dL_dq = grad(lambda q_: lagrangian_fn(q_, dq))(q)
#     dL_ddq = grad(lambda dq_: lagrangian_fn(q, dq_))(dq)

#     ddq_dt = np.zeros_like(dq)  # Use anp instead of np
#     for i in range(2):
#         dL_ddq_i_fn = lambda state_: grad(lambda dq_: lagrangian_fn(state_[:2], dq_))(state_[2:])[i]
#         ddq_dt[i] = dL_ddq_i_fn(state)

#     ddq = np.linalg.solve(np.eye(2), -dL_dq + ddq_dt)  # Use anp for matrix operations
#     return np.concatenate([dq, ddq])  # Use anp instead of np

# # --- TRAJECTORY SAMPLER ---
# def get_trajectory(dynamics_fn, t_span=[0, 3], timescale=15, y0=None, radius=None, noise_std=0.1, **kwargs):
#     t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))

#     if y0 is None:
#         y0 = np.random.rand(4) * 2.0 - 1.0
#     if radius is None:
#         radius = np.random.rand() + 1.3

#     y0 = y0 / np.sqrt((y0**2).sum()) * radius
#     ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)

#     y = ivp['y']
#     dydt = np.stack([dynamics_fn(None, y_) for y_ in y.T]).T

#     y += np.random.randn(*y.shape) * noise_std
#     return y, dydt, t_eval

# def get_dataset(dynamics_fn, seed=0, samples=50, test_split=0.5, **kwargs):
#     np.random.seed(seed)
#     xs, dxs = [], []

#     for _ in range(samples):
#         x, dx, _ = get_trajectory(dynamics_fn, **kwargs)
#         xs.append(x.T)
#         dxs.append(dx.T)

#     data = {'x': np.concatenate(xs), 'dx': np.concatenate(dxs).squeeze()}
#     split_ix = int(len(data['x']) * test_split)
#     split_data = {k: data[k][:split_ix] for k in ['x', 'dx']}
#     split_data.update({f'test_{k}': data[k][split_ix:] for k in ['x', 'dx']})
#     return split_data

