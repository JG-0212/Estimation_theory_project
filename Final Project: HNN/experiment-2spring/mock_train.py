# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import numpy as np
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from tqdm import tqdm
import wandb

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from hnn import HNN
from data import get_dataset
from utils import L2_loss, rk4
from torchdiffeq import odeint


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=2000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=20, type=int, help='number of gradient steps between prints')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout for model')
    parser.add_argument('--decay_param', default=1e-4, type=float, help='decay_param for model')
    parser.add_argument('--name', default='spring2', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--wand_runs', dest='wand_runs', action='store_true', help='used to log wandb metrics')
    parser.add_argument('--new_loss', dest='new_loss', action='store_true', help='train hnn with new updated loss function?')
    parser.add_argument('--improve', dest='improve', action='store_true', help='whether to improve the model with trajectory consistency')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def get_next_point_with_model(model_fwd, y0, dt):
    # y0: (B, 2)
    t0 = 0.0
    dy = rk4(model_fwd.time_derivative, y0, t0, dt)   # shape: (B, 2)
    return y0 + dy

def train(args):
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # init model and optimizer
    if args.verbose:
        print("Training baseline model:" if args.baseline else "Training HNN model:")

    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.dropout, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=args.decay_param)

    # arrange data
    data = get_dataset(seed=args.seed)
    x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32)
    test_x = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float32)
    dxdt = torch.Tensor(data['dx'])
    test_dxdt = torch.Tensor(data['test_dx'])
    
    # Setup for improve and new_loss options
    if args.new_loss:
        t_span, timescale = [0, 3], 35
        n_points = int(timescale * (t_span[1] - t_span[0]))
        kwargs = {
            't_eval': np.linspace(t_span[0], t_span[1], n_points),
            'rtol': 1e-12
        }
        batch_size = n_points
        
    if args.improve:
        # t_eval = np.linspace(0, 3, 35) # prone to change
        t_span, timescale = [0, 3], 35
        n_points = int(timescale * (t_span[1] - t_span[0]))
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        num_traj = int(len(x)/len(t_eval))
        traj_len = len(t_eval)
        assert num_traj == 50
        small_t_eval = t_eval[1]
        alpha = 0

    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in tqdm(range(args.total_steps+1)):
        optim.zero_grad()
        
        # train step
        dxdt_hat = model.rk4_time_derivative(x) if args.use_rk4 else model.time_derivative(x)
        loss = L2_loss(dxdt, dxdt_hat)
        
        if args.verbose and step % args.print_every == 0:
            print(f"Loss of term 1 {loss}")
        
        # Add improve loss if enabled
        if args.improve:
            next_pts = get_next_point_with_model(model, x, small_t_eval)
            next_pts[traj_len-1:-1:traj_len] = x[traj_len::traj_len]  # prevent penalization at end of trajectory
            extra_loss = L2_loss(x[1:], next_pts[:-1])
            loss += alpha * extra_loss
            
        # Add new_loss if enabled
        if args.new_loss:
            # Process in batches to avoid memory issues
            batch_loss = 0
            n_samples = len(data['x'])
            for i in range(0, n_samples, batch_size):
                batch_x0 = data['x'][i]
                hnn_path = integrate_model(model, t_span, batch_x0, **kwargs)
                hnn_x_hat = torch.tensor(hnn_path['y'].T, dtype=torch.float32)
                hnn_x = torch.tensor(data['x'][i:i+batch_size], dtype=torch.float32)
                batch_loss += L2_loss(hnn_x, hnn_x_hat)
            
            # Combine losses (you might want to weight them)
            loss += batch_loss / (n_samples)
            if args.verbose and step % args.print_every == 0:
                print(f"Loss of term 1 {batch_loss}")
        loss.backward()
        optim.step()

        # run test data
        test_dxdt_hat = model.rk4_time_derivative(test_x) if args.use_rk4 else model.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)
        
        # Add test improve loss if enabled
        if args.improve:
            test_next_pts = get_next_point_with_model(model, test_x, small_t_eval)
            test_next_pts[traj_len-1:-1:traj_len] = test_x[traj_len::traj_len]
            test_extra_loss = L2_loss(test_x[1:], test_next_pts[:-1])
            test_loss += test_extra_loss
            
        if args.wand_runs:
            wandb.log({
                "test_loss": test_loss.item(),
                "train_loss": loss.item()
            })
            
        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    train_dxdt_hat = model.time_derivative(x)
    train_dist = (dxdt - train_dxdt_hat)**2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat)**2
    
    if args.wand_runs:
        wandb.log({
            "final_train_loss": train_dist.mean().item(),
            "final_train_loss_std": train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            "final_test_loss": test_dist.mean().item(),
            "final_test_loss_std": test_dist.std().item()/np.sqrt(test_dist.shape[0])
        })
        
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
        .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
                test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

    return model, stats

def integrate_model(model, t_span, y0, **kwargs):
    
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,2)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

if __name__ == "__main__":
    args = get_args()
    # print(args.baseline)
    model, stats = train(args)

    if not args.wand_runs:
        os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
        label = '-baseline' if args.baseline else '-hnn'
        label = '-new_loss' + label if args.new_loss else label
        label = '-rk4' + label if args.use_rk4 else label
        label = '-improved' + label if args.improve else label
        path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)
        torch.save(model.state_dict(), path)