from neural_tangents.stax import Ksum, Tailor, Deriv1d

def kernels(model, d_eq, d_sl):
    equation = Deriv1d(model, d_eq, d_eq)
    sl_eq = Deriv1d(model, d_sl, d_eq)
    solution = Deriv1d(model, d_sl, d_sl)
    eq_sl = Deriv1d(model, d_eq, d_sl)
    _, _, kernel_dd = Tailor(equation, eq_sl, sl_eq, solution)
    _, _, kernel_td = Tailor(sl_eq, solution, ktd=True)
    _, _, kernel_tt = solution
    return kernel_dd, kernel_td, kernel_tt

def Gaussian_layer(layer, c=3):
    init_fn, apply_fn, kernel_fn = layer
    def new_kernel_fn(*args, **kwargs):
        if 'method' not in kwargs:
            return kernel_fn(*args, **kwargs, method='gaussian', c2=c**2)
        return kernel_fn(*args, **kwargs)
    return init_fn, apply_fn, new_kernel_fn

def IMQ_layer(layer, c=1):
    init_fn, apply_fn, kernel_fn = layer
    def new_kernel_fn(*args, **kwargs):
        if 'method' not in kwargs:
            return kernel_fn(*args, **kwargs, method='imq', c2=c**2)
        return kernel_fn(*args, **kwargs)
    return init_fn, apply_fn, new_kernel_fn

def FEM_layer(layer):
    init_fn, apply_fn, kernel_fn = layer
    def new_kernel_fn(*args, **kwargs):
        if 'method' not in kwargs:
            return kernel_fn(*args, **kwargs, method='fem')
        return kernel_fn(*args, **kwargs)
    return init_fn, apply_fn, new_kernel_fn

# Models for 1D Poission's Equation

def nn(std, d_eq=2, d_sl=0):
    model = stax.serial(
        stax.Dense(512, W_std=std, b_std=std),
        stax.Erf(),
        stax.Dense(  1, W_std=std, b_std=std),
    )
    return kernels(model, d_eq, d_sl)

def gaussian(c, d_eq=2, d_sl=0):
    model = Gaussian_layer(stax.Identity(), c=c)
    return kernels(model, d_eq, d_sl)

def imq(c, d_eq=2, d_sl=0):
    model = IMQ_layer(stax.Identity(), c=c)
    return kernels(model, d_eq, d_sl)

def fem(d_eq=2, d_sl=0):
    assert d_eq==2
    assert d_sl==0
    model = FEM_layer(stax.Identity())
    _, _, kernel_dd = model
    _, _, kernel_td = model
    return kernel_dd, kernel_td, None

tradition = [gaussian, imq, fem]

def source(x):
    return -5.4*x

def solve(x):
    return 1-0.9*x**3

def bound_for_fem(x):
    return 1-0.9*x

test_xs = np.linspace(0, 1, 200).reshape(-1, 1)
test_ys = solve(test_xs)
u_l2 = np.sqrt(np.square(test_ys).mean()).item()

class Experiments:
    def __init__(self, model, **model_kwargs):
        self.rmse = []
        self.amoc = []
        self.effc = []
        self.model = model
        self.is_traditional = model in tradition
        if model_kwargs or model is fem:
            self._kernels(**model_kwargs)
    def _kernels(self, **model_kwargs):
        self.kernel_dd, self.kernel_td, self.kernel_tt = self.model(**model_kwargs)
    def run(self, n_sample, t=None, diag_reg=0., compute_cov=False, precision=np.float32, **model_kwargs):
        train_eq = np.linspace(0, 1, n_sample)[1:-1].reshape(-1,1)
        train_lb = np.zeros((1, 1))
        train_rb = np.ones((1, 1))
        if self.model is fem:
            train_is = np.concatenate((train_lb, train_eq, train_rb))
            h = np.diff(np.sort(train_is, axis=None))
            M = np.diag(h[:-1]/3+h[1:]/3, 0)+np.diag(h[1:-1]/6, -1)+np.diag(h[1:-1]/6, 1)
            train_ts = M@source(train_eq)
        else:
            train_is = np.concatenate((train_eq, train_lb, train_rb))
            train_ts = np.concatenate((source(train_eq), solve(train_lb), solve(train_rb)))

        if model_kwargs:
            self._kernels(**model_kwargs)
        ensemble = nt.predict.gradient_descent_mse_ensemble(self.kernel_dd, train_is, train_ts, site=(-2,), diag_reg=diag_reg, precision=precision)        
        if t is not None:
            mean_out = ensemble.predict_fn(self.kernel_td, self.kernel_tt, x_test=test_xs, t=t, get='ntk', get_condition_number='ntk', iter_error=True)
            self.rmse = [i.item() for i in np.sqrt(np.mean((mean_out.reshape(-1, len(test_xs)))**2, axis=1))]
            #self.rmse = [i.item() for i in np.sqrt(np.mean((mean_out.reshape(-1, len(test_xs))-test_ys.reshape((1, -1)))**2, axis=1))]
            #print([i.item() for i in np.mean(np.diagonal(cov, axis1=1, axis2=2), axis=1)])
        if self.is_traditional:
            sl = ensemble.predict_fn(self.kernel_td, self.kernel_tt, x_test=test_xs, get='nngp', get_condition_number='nngp')
            mean_out = np.reshape(sl, (-1,))
            if self.model is fem:
                mean_out = -mean_out+bound_for_fem(test_xs.reshape(-1))
            self.amoc.append(ensemble.nngp_c[0])
            self.effc.append(ensemble.nngp_c[1])
        else:
            sl = ensemble.predict_fn(self.kernel_td, self.kernel_tt, x_test=test_xs, t=t, get_condition_number='ntk', compute_cov=compute_cov)
            if compute_cov:
                mean_out, cov_out = sl.ntk
                mean_out = np.reshape(mean_out, (-1,))
                std_out = np.sqrt(np.diag(cov_out))
                self.amoc.append(ensemble.ntk_c[0])
                self.effc.append(ensemble.ntk_c[1])
                self.rmse.append(np.sqrt(np.diag(cov_out).mean()).item())
            else:
                mean_out = np.reshape(sl.ntk, (-1,))
                self.amoc.append(ensemble.ntk_c[0])
                self.effc.append(ensemble.ntk_c[1])
        if t is None and not compute_cov:
            self.rmse.append(np.sqrt(np.square(mean_out-test_ys.reshape(-1)).mean()).item())
