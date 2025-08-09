import wandb
import torch
import os
from train import train, get_args

def sweepConfig():
    """Define the configuration for hyperparameter sweep focusing on dropout and decay_param"""
    config = {
        "method": "bayes",  # Bayesian optimization
        "metric": {"name": "final_test_loss", "goal": "minimize"},  # Optimization target
        "parameters": {
            "dropout": {"values": [0.0, 0.2, 0.4, 0.5]},
            "decay_param": {"values": [1e-5, 1e-4, 1e-3, 1e-2]}
        }
    }
    return config

def train_sweep():
    """Main training function for wandb sweep"""
    try:
        # Initialize wandb run
        wandb.init()
        config = wandb.config
        
        # Create descriptive run name
        run_name = f"dropout-{config.dropout:.3f}-decay-{config.decay_param:.2e}"
        wandb.run.name = run_name
        
        # Get default arguments and override with sweep parameters
        args = get_args()
        args.dropout = config.dropout
        args.decay_param = config.decay_param
        args.total_steps = 4
        args.wand_runs = True
        args.new_loss = True
        # Train model with current configuration
        model, stats = train(args)
        
    except Exception as e:
        print(f"[wandb/train] Error occurred: {e}")
        wandb.finish()  # Ensure wandb run is properly closed

def run_sweep():
    """Create and run the hyperparameter sweep"""
    # Create new sweep
    sweep_id = wandb.sweep(sweep=sweepConfig(), project="HNN-Hyperparameter-Tuning")
    # Run hyperparameter search
    print("Starting hyperparameter sweep for dropout and decay parameters")
    wandb.agent(sweep_id, function=train_sweep, count=3)  # 15 trials

if __name__ == "__main__":
    # Disable wandb code saving for cleaner runs
    os.environ["WANDB_DISABLE_CODE"] = "true"
    
    # Simply run the sweep when the file is executed
    run_sweep()