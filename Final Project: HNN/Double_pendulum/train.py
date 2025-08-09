# import torch, argparse
# import numpy as np
# import os, sys

# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PARENT_DIR)

# from nn_models import MLP
# from hnn import HNN
# from data import get_dataset
# from utils import L2_loss, rk4

# def get_args():
#     parser = argparse.ArgumentParser(description=None)
#     parser.add_argument('--input_dim', default=4, type=int, help='dimensionality of input tensor for double pendulum')
#     parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
#     parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
#     parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
#     parser.add_argument('--total_steps', default=2000, type=int, help='number of gradient steps')
#     parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
#     parser.add_argument('--name', default='double_pendulum', type=str, help='model name')
#     parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
#     parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
#     parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
#     parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
#     parser.add_argument('--seed', default=0, type=int, help='random seed')
#     parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
#     parser.set_defaults(feature=True)
#     return parser.parse_args()

# def train(args):
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
    
#     if args.verbose:
#         print("Training baseline model:" if args.baseline else "Training HNN model:")
    
#     output_dim = args.input_dim if args.baseline else 2
#     nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
#     model = HNN(args.input_dim, differentiable_model=nn_model,
#                 field_type=args.field_type, baseline=args.baseline)
#     optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)
    
#     data = get_dataset(seed=args.seed)
#     x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32)
#     test_x = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float32)
#     dxdt = torch.Tensor(data['dx'])
#     test_dxdt = torch.Tensor(data['test_dx'])
    
#     stats = {'train_loss': [], 'test_loss': []}
#     for step in range(args.total_steps + 1):
#         dxdt_hat = model.rk4_time_derivative(x) if args.use_rk4 else model.time_derivative(x)
#         loss = L2_loss(dxdt, dxdt_hat)
#         loss.backward()
#         optim.step()
#         optim.zero_grad()
        
#         test_dxdt_hat = model.rk4_time_derivative(test_x) if args.use_rk4 else model.time_derivative(test_x)
#         test_loss = L2_loss(test_dxdt, test_dxdt_hat)
        
#         stats['train_loss'].append(loss.item())
#         stats['test_loss'].append(test_loss.item())
#         if args.verbose and step % args.print_every == 0:
#             print(f"step {step}, train_loss {loss.item():.4e}, test_loss {test_loss.item():.4e}")
    
#     train_dxdt_hat = model.time_derivative(x)
#     train_dist = (dxdt - train_dxdt_hat) ** 2
#     test_dxdt_hat = model.time_derivative(test_x)
#     test_dist = (test_dxdt - test_dxdt_hat) ** 2
#     print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
#           .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
#                   test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))
    
#     return model, stats

# if __name__ == "__main__":
#     args = get_args()
#     model, stats = train(args)
    
#     os.makedirs(args.save_dir, exist_ok=True)
#     label = '-baseline' if args.baseline else '-hnn'
#     label = '-rk4' + label if args.use_rk4 else label
#     path = f'{args.save_dir}/{args.name}{label}.tar'
#     torch.save(model.state_dict(), path)
import torch, argparse
import numpy as np
import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from hnn import HNN
from hnn import LNN  # Import LNN here
from data import get_dataset, hamiltonian_dynamics, lagrangian_dynamics
from project_utils import L2_loss, rk4

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=4, type=int, help='dimensionality of input tensor for double pendulum')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=2000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='double_pendulum', type=str, help='model name')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--model_type', default='hnn', type=str, choices=['hnn', 'lnn'], help='Choose the model: hnn or lnn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.verbose:
        print("Training baseline model:" if args.baseline else f"Training {args.model_type.upper()} model:")

    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)

    if args.model_type == 'hnn':
        model = HNN(args.input_dim, differentiable_model=nn_model, field_type=args.field_type, baseline=args.baseline)
        dynamics_fn = hamiltonian_dynamics
    elif args.model_type == 'lnn':
        model = LNN(args.input_dim, differentiable_model=nn_model, field_type=args.field_type, baseline=args.baseline)
        dynamics_fn = lagrangian_dynamics

    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    data = get_dataset(seed=args.seed, dynamics_fn=dynamics_fn)
    x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32)
    test_x = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float32)
    dxdt = torch.Tensor(data['dx'])
    test_dxdt = torch.Tensor(data['test_dx'])

    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):
        dxdt_hat = model.rk4_time_derivative(x) if args.use_rk4 else model.time_derivative(x)
        loss = L2_loss(dxdt, dxdt_hat)
        loss.backward()
        optim.step()
        optim.zero_grad()

        test_dxdt_hat = model.rk4_time_derivative(test_x) if args.use_rk4 else model.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)

        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print(f"step {step}, train_loss {loss.item():.4e}, test_loss {test_loss.item():.4e}")

    train_dxdt_hat = model.time_derivative(x)
    train_dist = (dxdt - train_dxdt_hat) ** 2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat) ** 2
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                  test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))

    return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    os.makedirs(args.save_dir, exist_ok=True)
    label = '-baseline' if args.baseline else f'-{args.model_type}'
    label = '-rk4' + label if args.use_rk4 else label
    path = f'{args.save_dir}/{args.name}{label}.tar'
    torch.save(model.state_dict(), path)
