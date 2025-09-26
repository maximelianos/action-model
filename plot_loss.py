import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from loss_logger import LossLogger

def plot_training_curves(logger, args):
    """Plot training and validation curves"""
    print("Plotting training curves...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot train curve
    try:
        t, train_loss = logger.get("train")
        ax.plot(t, train_loss, 'b-', label='Training Loss', linewidth=2)
        print(f"Train average loss: {train_loss.mean():.6f}")
    except KeyError:
        print("No training loss data found")
    
    # Plot validation curve
    try:
        t, val_loss = logger.get("val")
        ax.plot(t, val_loss, 'r-', label='Validation Loss', linewidth=2)
        print(f"Val average loss: {val_loss.mean():.6f}")
    except KeyError:
        print("No validation loss data found")
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = Path(args.output_dir) / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")
    
    plt.close()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args : DictConfig):
    """Main execution function"""
    print("=== Transformer Autoencoder for Neural Time Series ===")

    # Load train log
    train_log = LossLogger(args.data_dir + "/loss.h5", overwrite=False)

    # Plot training curves
    plot_training_curves(train_log, args)

if __name__ == "__main__":
    main()
