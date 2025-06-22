import torch

class Test:
    def __init__(self, domain, pde, NN, num_eval_pts=201, dtype=torch.float32):        
        self.domain = domain
        self.pde = pde
        self.networks = NN
        self.num_eval_pts = num_eval_pts
        self.dtype = dtype

    def line(self):
        xmin, xmax = self.domain
        X_eval = torch.linspace(xmin, xmax, self.num_eval_pts, dtype=self.dtype)[:,None]
        return X_eval
    
    def rectangle(self):
        xmin, xmax, ymin, ymax = self.domain
        lin_x = torch.linspace(xmin, xmax, self.num_eval_pts, dtype=self.dtype)
        lin_y = torch.linspace(ymin, ymax, self.num_eval_pts, dtype=self.dtype)
        X, Y = torch.meshgrid(lin_x, lin_y, indexing='ij')
         
        X_eval = torch.vstack([X.ravel(), Y.ravel()]).T
        return X, Y, X_eval
    
    def Lshape(self):
        xmin, xmax, ymin, ymax = self.domain.tolist()  

        lin_x = torch.linspace(xmin, xmax, self.num_eval_pts,dtype=self.dtype)
        lin_y = torch.linspace(ymin, ymax, self.num_eval_pts,dtype=self.dtype)
        X, Y = torch.meshgrid(lin_x, lin_y, indexing='ij')

        x_mid = (xmin + xmax) / 2
        y_mid = (ymin + ymax) / 2
        x_mid = torch.tensor(x_mid, dtype=self.dtype)
        y_mid = torch.tensor(y_mid, dtype=self.dtype)

        mask = (X <= x_mid) | (Y >= y_mid)
        X_eval = torch.vstack([X[mask], Y[mask]]).T

        return X, Y, X_eval

    
    def cube(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.domain
        lin_x = torch.linspace(xmin, xmax, self.num_eval_pts, dtype=self.dtype)
        lin_y = torch.linspace(ymin, ymax, self.num_eval_pts, dtype=self.dtype)
        lin_z = torch.linspace(zmin, zmax, self.num_eval_pts, dtype=self.dtype)
        X, Y, Z = torch.meshgrid(lin_x, lin_y, lin_z, indexing='ij')

        X_eval = torch.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        return X, Y, Z, X_eval
    
    def time2D(self):
        xmin, xmax, ymin, ymax, tmin, tmax = self.domain
        lin_x = torch.linspace(xmin, xmax, self.num_eval_pts, dtype=self.dtype)
        lin_y = torch.linspace(ymin, ymax, self.num_eval_pts, dtype=self.dtype)
        lin_t = torch.linspace(tmin, tmax, self.num_eval_pts, dtype=self.dtype)
        X, Y, T = torch.meshgrid(lin_x, lin_y, lin_t, indexing='ij')

        X_eval = torch.vstack([X.ravel(), Y.ravel(), T.ravel()]).T

        return X, Y, T, X_eval

    def predict(self, X, coef_solution):
        self.networks.eval()    
        H_test = self.networks.get_last_hidden_layer(X)
        u_test = H_test @ coef_solution
        u_true = self.pde.exact(X)
        return u_test, u_true

    def error_compute(self, u_test, u_true):
        L2 = torch.sqrt(torch.sum((u_test - u_true) ** 2))
        Rele_L2 = L2 / torch.sqrt(torch.sum(u_true ** 2))
        Abs_mean = torch.mean(torch.abs(u_test - u_true))
        return L2, Rele_L2, Abs_mean