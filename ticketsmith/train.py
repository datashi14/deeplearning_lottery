import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import time
from ticketsmith.utils.config import load_config, hash_config
from ticketsmith.utils.artifacts import ArtifactManager
from ticketsmith.utils.data import get_mnist_loaders
from ticketsmith.models.mnist_cnn import MNISTCNN

from ticketsmith.models.unet import SimpleUNet
from ticketsmith.utils.diffusion_core import Diffusion

def get_model(config, device):
    family = config['model'].get('family', 'mnist_cnn')
    if family == 'mnist_cnn':
        return MNISTCNN().to(device)
    elif family == 'mnist_diffusion':
        return SimpleUNet().to(device)
    else:
        raise ValueError(f"Unknown model family: {family}")

def train_one_epoch(args, model, device, train_loader, optimizer, epoch, artifact_manager, config, post_step_callback=None):
    model.train()
    total_loss = 0
    correct = 0 # Only for classification
    
    family = config['model'].get('family', 'mnist_cnn')
    
    # Init Diffusion if needed
    if family == 'mnist_diffusion':
        # Create diffusion schedule on the fly or pass it? 
        # Typically fixed. Creating light object is fine.
        timesteps = config['training'].get('timesteps', 300)
        diffusion = Diffusion(timesteps=timesteps, device=device)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        if family == 'mnist_cnn':
            output = model(data)
            loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        elif family == 'mnist_diffusion':
            # Diffusion Training
            # Sample t uniformly
            t = torch.randint(0, diffusion.timesteps, (data.shape[0],), device=device).long()
            loss = diffusion.p_losses(model, data, t, loss_type="l2")
            # corrective metrics? MSE is loss.
        
        loss.backward()
        optimizer.step()
        
        if post_step_callback:
            post_step_callback(model)
        
        total_loss += loss.item() * data.size(0) 

    avg_loss = total_loss / len(train_loader.dataset)
    
    artifact_manager.log_metric('train_loss', avg_loss, epoch)
    print(f'Epoch {epoch} Train Loss: {avg_loss:.4f}')
    
    if family == 'mnist_cnn':
        accuracy = 100. * correct / len(train_loader.dataset)
        artifact_manager.log_metric('train_acc', accuracy, epoch)
        print(f'Accuracy: {accuracy:.2f}%')

def test(model, device, test_loader, artifact_manager, config, epoch=None):
    model.eval()
    test_loss = 0
    correct = 0
    
    family = config['model'].get('family', 'mnist_cnn')
    if family == 'mnist_diffusion':
        timesteps = config['training'].get('timesteps', 300)
        diffusion = Diffusion(timesteps=timesteps, device=device)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if family == 'mnist_cnn':
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            elif family == 'mnist_diffusion':
                # For validation loss, we just sample random t again?
                # Or fixed t for consistency? Random is fine for expectation.
                t = torch.randint(0, diffusion.timesteps, (data.shape[0],), device=device).long()
                loss = diffusion.p_losses(model, data, t, loss_type="l2") 
                test_loss += loss.item() * data.size(0)

    test_loss /= len(test_loader.dataset)
    
    acc_metric = 0.0
    if family == 'mnist_cnn':
        acc_metric = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc_metric:.2f}%)\n')
    else:
        print(f'Test set: Average loss: {test_loss:.4f} (Diffusion MSE)\n')

    if epoch is not None:
        artifact_manager.log_metric('val_loss', test_loss, epoch)
        if family == 'mnist_cnn':
            artifact_manager.log_metric('val_acc', acc_metric, epoch)
    return test_loss, acc_metric

def main():
    parser = argparse.ArgumentParser(description="Run dense training.")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    args = parser.parse_args()

    config = load_config(args.config)
    config_hash = hash_config(config)
    
    # Initialize Artifacts
    am = ArtifactManager()
    am.save_config(config)
    am.log_metric('config_hash', config_hash)
    
    print(f"Starting Dense Training")
    print(f"Config Hash: {config_hash}")
    
    # Setup Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Seed
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    
    # Load Data
    train_loader, val_loader = get_mnist_loaders(
        batch_size=config['training'].get('batch_size', 64)
    )
    
    # Model
    model = get_model(config, device)
    
    # Optimization
    optimizer = optim.SGD(model.parameters(), 
                          lr=config['training'].get('lr', 0.01),
                          momentum=config['training'].get('momentum', 0.9))
    
    # Training Loop
    epochs = config['training'].get('epochs', 5)
    
    for epoch in range(1, epochs + 1):
        train_one_epoch(args, model, device, train_loader, optimizer, epoch, am, config)
        test(model, device, val_loader, am, config, epoch)
        
    # Save Final Model
    am.save_checkpoint(model.state_dict(), 'final_dense')
    
    am.save_metrics()
    print(f"Training complete. Artifacts saved to {am.run_dir}")


if __name__ == "__main__":
    main()
