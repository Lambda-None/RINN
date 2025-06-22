import torch
import torch.nn as nn
from torch.autograd.functional import jvp

class SinActivation(nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(torch.pi * x)

class CosActivation(nn.Module):
    def __init__(self):
        super(CosActivation, self).__init__()

    def forward(self, x):
        return torch.cos(torch.pi * x)
    
class GaussActivation(nn.Module):
    def __init__(self):
        super(GaussActivation, self).__init__()

    def forward(self, x):
        return torch.exp(- x**2)

class PIELM(nn.Module):
    def __init__(self, mlp_layers, act='tanh', w_init='xavier_normal'):
        super(PIELM, self).__init__()
        self.layers = mlp_layers
        self.activation = act
        self.weight_init = w_init
        self.model = nn.Sequential()

        # 构建完整的隐藏层部分（含激活）
        self.hidden_model = nn.Sequential()
        for i in range(len(mlp_layers) - 2):  
            lin = nn.Linear(mlp_layers[i], mlp_layers[i+1], bias=True)
            self.init_weights(lin)
            self.hidden_model.add_module(f"hidden_linear_{i+1}", lin)
            self.hidden_model.add_module(f"hidden_activation_{i+1}", self.get_activation())

        # 构建输出层，只做线性变换，且无偏置
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

    def get_last_hidden_layer(self, X):
        return self.hidden_model(X)

    def forward(self, X):
        hidden_out = self.hidden_model(X)
        output = self.output_model(hidden_out)
        return output
    
    def compute_gradients(self, X):
        dim = X.size(-1)
        assert dim in {1, 2, 3}, f"仅支持1D/2D/3D输入,当前维度为{dim}"
        
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


