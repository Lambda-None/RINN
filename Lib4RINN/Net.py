import torch
import torch.nn as nn
from torch.autograd.functional import jvp

class SinActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi * x)

class CosActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(torch.pi * x)

class GaussActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(- x**2)

class PIELM(nn.Module):
    def __init__(self, mlp_layers, act='tanh', w_init='uniform', b_init=None):
        super(PIELM, self).__init__()
        self.layers = mlp_layers
        self.activation = act
        self.weight_init = w_init
        self.bias_init = b_init
        self.model = nn.Sequential()

        self.hidden_model = nn.Sequential()
        for i in range(len(mlp_layers) - 2):  
            lin = nn.Linear(mlp_layers[i], mlp_layers[i+1], bias=True)
            self.init_weights(lin)
            self.init_bias(lin)
            self.hidden_model.add_module(f"hidden_linear_{i+1}", lin)
            self.hidden_model.add_module(f"hidden_activation_{i+1}", self.get_activation())

        input_dim = mlp_layers[-2]
        output_dim = mlp_layers[-1]
        self.output_model = nn.Linear(input_dim, output_dim, bias=False)
        self.init_weights(self.output_model)  

    def init_weights(self, layer):
        if '_' in self.weight_init:
            method_name, param_str = self.weight_init.rsplit('_', 1)
            param = float(param_str)
        else:
            method_name = self.weight_init
            param = None  
        init_methods = {
            'normal': lambda x, p: nn.init.normal_(x.weight, mean=0.0, std=p or 1.0),
            'uniform': lambda x, p: nn.init.uniform_(x.weight, a=-p, b=p),
            'xavier_uniform': lambda x, p: nn.init.xavier_uniform_(x.weight, gain=p or 1.0),
            'kaiming_uniform': lambda x, p: nn.init.kaiming_uniform_(x.weight, a=p or 1.0),
        }
        if method_name not in init_methods:
            raise ValueError(f"Unsupported weight initialization method: {method_name}")
        init_methods[method_name](layer, param)

    def init_bias(self, layer):
        if not hasattr(layer, "bias") or layer.bias is None:
            return

        if not self.bias_init:
            return   

        if "_" in self.bias_init:
            method_name, param = self.bias_init.rsplit("_", 1)
            param = float(param)
        else:
            method_name = self.bias_init
            param = None

        init_methods = {
            "uniform": lambda b, p: nn.init.uniform_(b, a=-(p or 1.0), b=(p or 1.0)),
            "random":  lambda b, p: nn.init.normal_(b, mean=0.0, std=(p or 1.0)),
            "zeros":   lambda b, p: nn.init.constant_(b, 0.0),
            "ones":    lambda b, p: nn.init.constant_(b, 1.0),
        }

        if method_name not in init_methods:
            raise ValueError(f"Unsupported bias initialization method: {method_name}")

        init_methods[method_name](layer.bias, param)

    def get_activation(self):
        activations = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'sin': SinActivation(),
            'cos': CosActivation(),
            'gauss': GaussActivation()
        }
        return activations[self.activation]

    def get_last_hidden_layer(self, X):
        return self.hidden_model(X)

    def forward(self, X):
        hidden_out = self.hidden_model(X)
        output = self.output_model(hidden_out)
        return output
    
    def compute_gradients(self, X):
        dim = X.size(-1)
        assert dim in {1, 2, 3}, f"only support 1D/2D/3D inputs, current dimension is{dim}"
        
        directions = [torch.zeros_like(X) for _ in range(dim)]
        for i in range(dim):
            directions[i][..., i] = 1.0

        def f(x):
            return self.get_last_hidden_layer(x)

        first_derivs = []
        for v in directions:
            _, deriv = jvp(f, X, v, create_graph=True)
            first_derivs.append(deriv.detach())

        second_derivs = []
        for v in directions:
            df = lambda x: jvp(f, x, v, create_graph=True)[1]
            _, deriv2 = jvp(df, X, v)
            second_derivs.append(deriv2.detach())

        return (*first_derivs, *second_derivs)
    

    def compute_gradients_up_to_3(self, X):
        dim = X.shape[-1]
        assert dim in {1, 2, 3}, f"only support 1D/2D/3D inputs, current dimension is{dim}"

        directions = [torch.zeros_like(X) for _ in range(dim)]
        for i in range(dim):
            directions[i][..., i] = 1.0

        f = lambda x: self.get_last_hidden_layer(x)

        first_derivs = []
        second_derivs = []
        third_derivs = []

        for v in directions:
            _, j1 = torch.autograd.functional.jvp(f, X, v, create_graph=True)
            first_derivs.append(j1.detach())

            df = lambda x: torch.autograd.functional.jvp(f, x, v, create_graph=True)[1]
            _, j2 = torch.autograd.functional.jvp(df, X, v)
            second_derivs.append(j2.detach())

            d2f = lambda x: torch.autograd.functional.jvp(df, x, v, create_graph=True)[1]
            _, j3 = torch.autograd.functional.jvp(d2f, X, v)
            third_derivs.append(j3.detach())

        return (*first_derivs, *second_derivs, *third_derivs)


class Hard_PIELM(nn.Module):
    def __init__(self, mlp_layers, act='tanh', w_init='xavier_normal', domain=None):
        '''
        The class is only compatible with one hidden layer, and need to use the hard-bounday with square(hyper-) domain
        '''
        super(Hard_PIELM, self).__init__()
        self.layers = mlp_layers
        self.activation = act
        self.weight_init = w_init
        self.domain = domain
        self.model = nn.Sequential()

        self.hidden_model = nn.Sequential()
        for i in range(len(mlp_layers) - 2):  
            lin = nn.Linear(mlp_layers[i], mlp_layers[i+1], bias=True)
            self.init_weights(lin)
            self.hidden_model.add_module(f"hidden_linear_{i+1}", lin)
            self.hidden_model.add_module(f"hidden_activation_{i+1}", self.get_activation())

        input_dim = mlp_layers[-2]
        output_dim = mlp_layers[-1]
        self.output_model = nn.Linear(input_dim, output_dim, bias=False)
        self.init_weights(self.output_model)  

    def init_weights(self, layer):
        if '_' in self.weight_init:
            method_name, param_str = self.weight_init.rsplit('_', 1)
            param = float(param_str)
        else:
            method_name = self.weight_init
            param = None  
        init_methods = {
            'normal': lambda x, p: nn.init.normal_(x.weight, mean=0.0, std=p or 1.0),
            'uniform': lambda x, p: nn.init.uniform_(x.weight, a=-p, b=p),
            'xavier_uniform': lambda x, p: nn.init.xavier_uniform_(x.weight, gain=p or 1.0),
            'kaiming_uniform': lambda x, p: nn.init.kaiming_uniform_(x.weight, a=p or 1.0),
        }
        if method_name not in init_methods:
            raise ValueError(f"Unsupported initialization method: {method_name}")
        init_methods[method_name](layer, param)


    def get_activation(self):
        activations = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'sin': SinActivation(),
            'cos': CosActivation(),
            'gauss': GaussActivation()
        }
        return activations[self.activation]
    
    def distance_function(self, X):
        d = X.shape[1]
        dom = self.domain.to(device=X.device, dtype=X.dtype).reshape(d, 2)
        a = dom[:, 0].unsqueeze(0)
        b = dom[:, 1].unsqueeze(0)
        return ((X - a) * (b - X)).prod(dim=1, keepdim=True)

    def get_last_hidden_layer(self, X):
        f_d = self.distance_function(X)
        hidden_output = self.hidden_model(X)
        return f_d * hidden_output

    def forward(self, X):
        hidden_out = self.get_last_hidden_layer(X)
        output = self.output_model(hidden_out)
        return output
    
    def _activation_derivatives(self, z):
        act = self.activation.lower()
        if act == 'tanh':
            phi = torch.tanh(z)
            phi_p = 1.0 - phi * phi
            phi_pp = -2.0 * phi * (1.0 - phi * phi)
        elif act == 'sigmoid':
            phi = torch.sigmoid(z)
            phi_p = phi * (1.0 - phi)
            phi_pp = phi_p * (1.0 - 2.0 * phi)
        elif act == 'sin':
            phi = torch.sin(z)
            phi_p = torch.cos(z)
            phi_pp = -phi
        elif act == 'cos':
            phi = torch.cos(z)
            phi_p = -torch.sin(z)
            phi_pp = -phi
        elif act == 'gauss':
            phi = torch.exp(-z * z)
            phi_p = -2.0 * z * phi
            phi_pp = (-2.0 + 4.0 * z * z) * phi
        else:
            raise ValueError(f"Unsupported activation for analytic derivatives: {self.activation}")
        return phi, phi_p, phi_pp


    def laplacian_last_hidden(self, X, eps: float = 1e-12):
        if len(self.layers) != 3:
            raise RuntimeError("laplacian_last_hidden is only compatible with one hidden layer(len(mlp_layers)==3)。")

        if not torch.is_tensor(X):
            X = torch.as_tensor(X, device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)

        N, d = X.shape
        
        lin = None
        for m in self.hidden_model:
            if isinstance(m, torch.nn.Linear):
                lin = m
                break
        if lin is None:
            raise RuntimeError("Hidden linear layer not found.")

        W = lin.weight        # (H, d)
        b = lin.bias
        H = W.shape[0]

        z = X @ W.t()
        if b is not None:
            z = z + b.view(1, -1)

        phi, phi_p, phi_pp = self._activation_derivatives(z)

        dom = self.domain.to(device=X.device, dtype=X.dtype).reshape(d, 2)
        a = dom[:, 0].view(1, d)
        b_dom = dom[:, 1].view(1, d)

        g = (X - a) * (b_dom - X)
        g = torch.clamp(g, min=eps)
        gprime = (a + b_dom) - 2.0 * X

        f_d = torch.prod(g, dim=1, keepdim=True)  # (N,1)

        w_norm_sq = (W * W).sum(dim=1).view(1, H)  # (1,H)
        ratio = gprime / g                          # (N,d)
        S = ratio @ W.t()                           # (N,H)
        sum_inv_g = torch.sum(1.0 / g, dim=1, keepdim=True)  # (N,1)

        term1 = phi_pp * w_norm_sq
        term2 = 2.0 * phi_p * S
        term3 = -2.0 * phi * sum_inv_g

        lap = f_d * (term1 + term2 + term3)  # (N,H)
        return lap