import torch
import numpy as np
from scipy.stats import qmc


class BcsPoint:
    @staticmethod
    def line(domain, num_bdry_pts, dtype=torch.float32):
        ones = torch.ones((num_bdry_pts, 1), dtype=dtype)
        
        xmin, xmax = domain
        
        X_bcs = torch.vstack([
            (xmin * ones),  # 左边界
            (xmax * ones),  # 右边界
        ])
        
        X_ini = torch.empty([0, len(domain) // 2], dtype=dtype)  # you must have this as a placeholder
        
        return X_ini, X_bcs
    
    @staticmethod
    def rectangle(domain, num_bdry_pts, dtype=torch.float32):
        ones = torch.ones((num_bdry_pts, 1), dtype=dtype)
        
        xmin, xmax, ymin, ymax = domain
        lin_y = torch.linspace(ymin, ymax, num_bdry_pts, dtype=dtype)[:, None]
        lin_x = torch.linspace(xmin, xmax, num_bdry_pts, dtype=dtype)[:, None]
        
        X_bcs = torch.vstack([
            torch.hstack((xmin * ones, lin_y)),   # left
            torch.hstack((xmax * ones, lin_y)),   # right
            torch.hstack((lin_x, ymin * ones)),   # bottom
            torch.hstack((lin_x, ymax * ones))    # top
        ])
        
        X_ini = torch.empty([0, len(domain) // 2], dtype=dtype)  # you must have this as a placeholder
        
        return X_ini, X_bcs
    
    @staticmethod
    def Lshape(domain, num_bdry_pts, dtype=torch.float32):
        ones_full = torch.ones((num_bdry_pts, 1), dtype=dtype)           # 用于整条边
        ones_half = torch.ones((num_bdry_pts // 2, 1), dtype=dtype)      # 用于一半长度的边
        
        xmin, xmax, ymin, ymax = domain
        
        lin_x = torch.linspace(xmin, xmax, num_bdry_pts, dtype=dtype)[:, None]
        lin_y = torch.linspace(ymin, ymax, num_bdry_pts, dtype=dtype)[:, None]       
        lin_x_left = torch.linspace(xmin, (xmin + xmax) / 2, num_bdry_pts // 2, dtype=dtype)[:, None]
        lin_x_right = torch.linspace((xmin + xmax) / 2, xmax, num_bdry_pts // 2, dtype=dtype)[:, None]      
        lin_y_left = torch.linspace(ymin, (ymin + ymax) / 2, num_bdry_pts // 2, dtype=dtype)[:, None]
        lin_y_right = torch.linspace((ymin + ymax) / 2, ymax, num_bdry_pts // 2, dtype=dtype)[:, None]
        
        X_bcs = torch.vstack([
                # left 整条边
                torch.hstack((xmin * ones_full, lin_y)),                       
                # right_bottom 半条边
                torch.hstack((xmax * ones_half, lin_y_right)),                 
                # right_top    半条边
                torch.hstack(((xmin + xmax) / 2 * ones_half, lin_y_left)),     
                # bottom_left  半条边
                torch.hstack((lin_x_left, ymin * ones_half)),                  
                # bottom_right 半条边
                torch.hstack((lin_x_right, (ymin + ymax) / 2 * ones_half)),    
                # top 整条边
                torch.hstack((lin_x, ymax * ones_full))                        
            ])
        X_ini = torch.empty([0, len(domain) // 2], dtype=dtype)  # you must have this as a placeholder
        
        return X_ini, X_bcs
    
    @staticmethod
    def triangle(domain, num_bdry_pts, dtype=torch.float32):
        xmin, xmax, ymin, ymax = domain
        pt1 = torch.array([xmin, ymin], dtype=dtype)
        pt2 = torch.array([xmin, ymax], dtype=dtype)
        pt3 = torch.array([xmax, ymin], dtype=dtype)
        pt4 = torch.array([xmin, ymax], dtype=dtype)
        pt5 = torch.array([xmax, ymax], dtype=dtype)
        
        X_bcs = torch.vstack([
            torch.linspace(pt1, pt2, num_bdry_pts),
            torch.linspace(pt1, pt3, num_bdry_pts),
            torch.linspace(pt4, pt5, num_bdry_pts)
        ]) 
           
        X_ini = torch.empty([0, len(domain) // 2], dtype=dtype)  # you must have this as a placeholder
        
        return X_ini, X_bcs
    
    @staticmethod
    def cube(domain, num_bdry_pts, dtype=torch.float32):
        xmin, xmax, ymin, ymax, zmin, zmax = domain
        
        samples = []
    
        # 生成x的两个面（x_min和x_max）
        for x_val in [xmin, xmax]:
            y = (ymax - ymin) * torch.rand(num_bdry_pts, dtype=dtype) + ymin
            z = (zmax - zmin) * torch.rand(num_bdry_pts, dtype=dtype) + zmin
            x = torch.full((num_bdry_pts,), x_val, dtype=dtype)
            samples.append(torch.stack([x, y, z], dim=1))
        
        # 生成y的两个面（y_min和y_max）
        for y_val in [ymin, ymax]:
            x = (xmax - xmin) * torch.rand(num_bdry_pts, dtype=dtype) + xmin
            z = (zmax - zmin) * torch.rand(num_bdry_pts, dtype=dtype) + zmin
            y = torch.full((num_bdry_pts,), y_val, dtype=dtype)
            samples.append(torch.stack([x, y, z], dim=1))
        
        # 生成z的两个面（z_min和z_max）
        for z_val in [zmin, zmax]:
            x = (xmax - xmin) * torch.rand(num_bdry_pts, dtype=dtype) + xmin
            y = (ymax - ymin) * torch.rand(num_bdry_pts, dtype=dtype) + ymin
            z = torch.full((num_bdry_pts,), z_val, dtype=dtype)
            samples.append(torch.stack([x, y, z], dim=1))

        X_bcs = torch.cat(samples, dim=0)
           
        X_ini = torch.empty([0, len(domain) // 2], dtype=dtype)  # you must have this as a placeholder
        
        return X_ini, X_bcs

    
    
class IniBcsPoint:
    @staticmethod
    def rectangle(domain, num_init_pts, num_bdry_pts, dtype=torch.float32):
        ones_init = torch.ones((num_init_pts, 1), dtype=dtype)
        ones_bdry = torch.ones((num_bdry_pts, 1), dtype=dtype)
        
        xmin, xmax, tmin, tmax = domain
        
        # 初始条件点
        x_vals = torch.linspace(xmin, xmax, num_init_pts, dtype=dtype)[:, None]
        X_ini = torch.hstack((x_vals, tmin * ones_init))
        
        # 边界条件点
        lin_t = torch.linspace(tmin, tmax, num_bdry_pts, dtype=dtype)[:, None]        
        X_bcs = torch.vstack([
            torch.hstack((xmin * ones_bdry, lin_t)),
            torch.hstack((xmax * ones_bdry, lin_t))
        ])
        
        return X_ini, X_bcs
    
    @staticmethod
    def cube(domain, num_init_pts, num_bdry_pts, dtype=torch.float32):
        ones_init = torch.ones((num_init_pts, 1), dtype=dtype)
        ones_bdry = torch.ones((num_bdry_pts, 1), dtype=dtype)

        xmin, xmax, ymin, ymax, tmin, tmax = domain

        ### 初始条件点
        x_ini = (xmax-xmin) * torch.rand((num_init_pts,1), dtype=dtype) + xmin
        y_ini = (ymax-ymin) * torch.rand((num_init_pts,1), dtype=dtype) + ymin
        X_ini = torch.cat([x_ini, y_ini, tmin*ones_init], dim=1)

        ### 边界条件点
        x_bcs = (xmax-xmin) * torch.rand((num_bdry_pts,1), dtype=dtype) + xmin
        y_bcs = (ymax-ymin) * torch.rand((num_bdry_pts,1), dtype=dtype) + ymin
        t_vals = (tmax-tmin) * torch.rand((num_bdry_pts,1), dtype=dtype) + tmin

        X_bcs = torch.vstack([
            torch.hstack((x_bcs, ymin*ones_bdry, t_vals)), 
            torch.hstack((x_bcs, ymax*ones_bdry, t_vals)), 
            torch.hstack((xmin*ones_bdry, y_bcs, t_vals)), 
            torch.hstack((xmax*ones_bdry, y_bcs, t_vals))
            ])

        return X_ini, X_bcs
    
    
class ResPoint:
    @staticmethod
    def get_sampling_function(sampling_method="sobol"):
        sampling_methods = {
            "uniform": sample_uniform,
            "random": sample_random,
            "sobol": sample_sobel,
            "latin_hypercube": sample_LatinHypercube
        }
        return sampling_methods[sampling_method]

    @staticmethod
    def line(domain, num_colloc_pts, sampling_method="sobol", dtype=torch.float32):
        sampling_function = ResPoint.get_sampling_function(sampling_method)
        X_res = sampling_function(domain, num_colloc_pts, dtype)
        return X_res
    
    @staticmethod
    def rectangle(domain, num_colloc_pts, sampling_method="sobol", dtype=torch.float32):
        sampling_function = ResPoint.get_sampling_function(sampling_method)
        X_res = sampling_function(domain, num_colloc_pts, dtype)
        return X_res
    
    @staticmethod
    def cube(domain, num_colloc_pts, sampling_method="sobol", dtype=torch.float32):
        sampling_function = ResPoint.get_sampling_function(sampling_method)
        X_res = sampling_function(domain, num_colloc_pts, dtype)
        return X_res
    
    @staticmethod
    def Lshape(domain, num_colloc_pts, sampling_method="sobol", dtype=torch.float32):
        sampling_function = ResPoint.get_sampling_function(sampling_method)
        X_res = sampling_function(domain, num_colloc_pts, dtype)
        xmin, xmax, ymin, ymax = domain
        mask = ~((X_res[:, 0] >= (xmin + xmax) / 2) & (X_res[:, 1] <= (ymin + ymax) / 2))
        X_res = X_res[mask]
        return X_res
    
    @staticmethod
    def triangle(domain, num_colloc_pts, sampling_method="sobol", dtype=torch.float32):
        sampling_function = ResPoint.get_sampling_function(sampling_method)
        X_res = sampling_function(domain, num_colloc_pts, dtype)
        xmin, xmax, ymin, ymax = domain
        mask = (X_res[:, 1] < torch.round((ymin - ymax) / (xmax - xmin) * (X_res[:, 0] - xmin) + ymax, 5))
        X_res = X_res[mask]
        return X_res


def sample_uniform(domain, num_colloc_pts, dtype=torch.float32):
    dims = len(domain) // 2
    grid_points = int(torch.ceil(num_colloc_pts ** (1/dims)))
    
    # Generate evenly spaced grid points for each dimension
    axes = [torch.linspace(domain[2*i], domain[2*i+1], grid_points, dtype=dtype) for i in range(dims)]
    mesh = torch.meshgrid(*axes)
    X = torch.stack([m.flatten() for m in mesh], dim=1)
    return X

def sample_random(domain, num_colloc_pts, dtype=torch.float32):
    dims = len(domain) // 2
    
    # Generate random samples
    X = torch.rand((num_colloc_pts, dims), dtype=dtype)
    
    # Rescale to the specified domain (without loops)
    xmin = domain[::2]
    xmax = domain[1::2]
    X = xmin + (xmax - xmin) * X  # Vectorized rescaling
    return X

def sample_sobel(domain, num_colloc_pts, dtype=torch.float32):
    dims = len(domain) // 2
    sampler = qmc.Sobol(d=dims, scramble=False)
    X = sampler.random(n=num_colloc_pts)
    X = torch.tensor(X, dtype=dtype)
    
    # Rescale to the specified domain (without loops)
    xmin = domain[::2]  # Extracts all xmin values for each dimension
    xmax = domain[1::2]  # Extracts all xmax values for each dimension
    X = xmin + (xmax - xmin) * X  # Vectorized rescaling
    return X

def sample_LatinHypercube(domain, num_colloc_pts, dtype=torch.float32):
    dims = len(domain) // 2 
    sampler = qmc.LatinHypercube(d=dims)
    
    # 使用 NumPy dtype 生成随机点
    np_dtype = {
        torch.float32: np.float32,
        torch.float64: np.float64
    }[dtype]
    
    points = sampler.random(n=num_colloc_pts).astype(np_dtype)
    
    # Rescale to the specified domain (vectorized)
    xmin = np.array(domain[::2], dtype=np_dtype)
    xmax = np.array(domain[1::2], dtype=np_dtype)
    X_np = xmin + (xmax - xmin) * points

    # 转为 PyTorch tensor
    X = torch.from_numpy(X_np).to(dtype)
    return X