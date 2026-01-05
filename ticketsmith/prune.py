import argparse
import torch
import torch.optim as optim
import copy
import sys
import os
from ticketsmith.utils.config import load_config, hash_config
from ticketsmith.utils.artifacts import ArtifactManager
from ticketsmith.utils.data import get_mnist_loaders
from ticketsmith.models.mnist_cnn import MNISTCNN
from ticketsmith.models.unet import SimpleUNet
from ticketsmith.utils.diffusion_core import Diffusion
from ticketsmith.utils.pruning import Pruner
from ticketsmith.utils.quality import QualityGate, generate_sample_grid
from ticketsmith.utils.benchmark import run_benchmark
# Import training helpers
from ticketsmith.train import train_one_epoch, test, get_model
import time

def main():
    parser = argparse.ArgumentParser(description="Run pruning and retraining.")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    args = parser.parse_args()

    config = load_config(args.config)
    config_hash = hash_config(config)
    
    am = ArtifactManager()
    am.save_config(config)
    am.log_metric('config_hash', config_hash)
    
    print(f"Starting Prune and Retrain (IMP)")
    print(f"Config Hash: {config_hash}")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader = get_mnist_loaders(
        batch_size=config['training'].get('batch_size', 64)
    )
    
    # Model Init
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    
    model = get_model(config, device)
    
    # Save Initial State (theta_0)
    initial_state = copy.deepcopy(model.state_dict())
    
    # Pruning Config
    rounds = config.get('pruning', {}).get('rounds', 3)
    prune_rate = config.get('pruning', {}).get('rate', 0.2)
    
    pruner = Pruner(model, pruning_rate=prune_rate)
    current_masks = None
    
    family = config['model'].get('family', 'mnist_cnn')

    for round_idx in range(rounds + 1):
        print(f"\n--- Round {round_idx} / {rounds} ---")
        sparsity = 0.0
        
        # 1. Setup Model
        if round_idx == 0:
            # Round 0: Dense training
            pass
        else:
            # Round > 0: Prune -> Rewind -> Retrain
            print("Pruning...")
            # Compute masks from the trained model of previous round
            current_masks, sparsity_stats = pruner.compute_mask(current_masks)
            
            am.log_metric(f'round_{round_idx}_sparsity', sparsity_stats['global_sparsity'])
            am.log_metric('sparsity_level', sparsity_stats['global_sparsity'])
            
            sparsity = sparsity_stats['global_sparsity']
            
            # Rewind
            print("Rewinding to theta_0...")
            model.load_state_dict(initial_state)
            
            # Apply Initial Masking (zero out weights before starting)
            pruner.apply_mask(current_masks)
            
        
        # 2. Train
        # Re-init optimizer each round
        optimizer = optim.SGD(model.parameters(), 
                              lr=config['training'].get('lr', 0.01),
                              momentum=config['training'].get('momentum', 0.9))
        
        # Callback to enforce mask
        def enforce_mask_callback(mdl):
            if current_masks:
                for name, module in mdl.named_modules():
                    if name in current_masks:
                        mask = current_masks[name].to(module.weight.device)
                        module.weight.data *= mask
        
        callback = enforce_mask_callback if current_masks else None
        
        epochs = config['training'].get('epochs', 5)
        
        for epoch in range(1, epochs + 1):
            train_one_epoch(args, model, device, train_loader, optimizer, epoch, am, config, post_step_callback=callback)
            
            loss, metric_score = test(model, device, val_loader, am, config, epoch=None) 
            
            am.log_metric(f'round_{round_idx}_loss', loss, epoch)
             
            if family == 'mnist_cnn':
                am.log_metric(f'round_{round_idx}_acc', metric_score, epoch)
                am.log_metric('global_acc', metric_score) 
            elif family == 'mnist_diffusion':
                # metric_score is meaningless here (0)
                # We need "Accuracy" or "Quality" for the report.
                # Just log scalar 0 or use loss?
                # The report logic expects 'Accuracy' column.
                # Maybe map 1/loss or just negative loss? 
                # Or run the classifier?
                pass
            
        # 3. Save Round Checkpoint
        am.save_checkpoint(model.state_dict(), f'round_{round_idx}_model')
        
        # 4. Snapshot current performance for Quality Gate
        current_metrics = {
            'val_loss': am.metrics.get(f'round_{round_idx}_loss')[-1]['value']
        }
        if family == 'mnist_cnn':
             current_metrics['val_acc'] = am.metrics.get(f'round_{round_idx}_acc')[-1]['value']
        else:
             # Diffusion: Use loss as proxy if no classifier score
             # Ideally we run classifier here.
             pass
        
        if round_idx == 0:
            baseline_metrics = current_metrics
        
        # 5. Run Quality Gate
        qg = QualityGate(config)
        passed, reasons, signals = qg.evaluate(current_metrics, baseline_metrics)
        
        gate_result = {
            'passed': passed,
            'reasons': reasons,
            'signals': signals,
            'thresholds': qg.config
        }
        am.log_metric(f'round_{round_idx}_gate', gate_result)
        print(f"Quality Gate Round {round_idx}: {'PASS' if passed else 'FAIL'} {reasons}")
        
        # 6. Generate Sample Grid
        # Works for both families now
        grid_path = os.path.join(am.run_dir, 'plots', f'grid_round_{round_idx}.png')
        generate_sample_grid(model, device, val_loader, grid_path, config=config)
        
        # 7. Run Benchmark (Serving Awareness)
        bench_variant_name = f"round_{round_idx}_sparsity_{int(sparsity*100)}"
        run_benchmark(model, config, device, am, variant_name=bench_variant_name)

        # 8. Random Re-init Comparison
        if round_idx > 0 and config.get('pruning', {}).get('compare_random', False):
            print(f"--- Round {round_idx} Random Re-init Comparison ---")
            rand_model = get_model(config, device) # Dynamic
            
            # Apply initial mask (zero out)
            for name, module in rand_model.named_modules():
                if name in current_masks:
                    mask = current_masks[name].to(device)
                    module.weight.data *= mask
                    
            rand_optimizer = torch.optim.SGD(rand_model.parameters(), 
                              lr=config['training'].get('lr', 0.01),
                              momentum=config['training'].get('momentum', 0.9))
            
            def enforce_mask_rand(mdl):
                for name, module in mdl.named_modules():
                    if name in current_masks:
                        mask = current_masks[name].to(module.weight.device)
                        module.weight.data *= mask
                        
            for epoch in range(1, epochs + 1):
                train_one_epoch(args, rand_model, device, train_loader, rand_optimizer, epoch, am, config, post_step_callback=enforce_mask_rand)
                loss, score = test(rand_model, device, val_loader, am, config, epoch=None)
                if family == 'mnist_cnn':
                     am.log_metric(f'round_{round_idx}_random_acc', score, epoch)
                
            print(f"Random Re-init Round {round_idx} Score: {score:.2f}")

    am.save_metrics()
    print(f"IMP complete. Artifacts saved to {am.run_dir}")

if __name__ == "__main__":
    main()
