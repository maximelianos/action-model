import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

from loss_logger import LossLogger

def plot_training_curves(train_log, args):
    """Plot training and validation loss curves"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # plot train curve
    t, train_loss = train_log.get("train")
    ax.plot(t, train_loss, 'b-', label='Training Loss')

    # plot val curve
    t, val_loss = train_log.get("val")
    ax.plot(t, val_loss, 'r-', label='Validation Loss')
    
    print("train average loss %.3f" % train_loss.mean())
    print("val average loss %.3f" % val_loss.mean())

    # set ylim
    ylim = max(train_loss.mean(), val_loss.mean())
    ax.set_ylim((0, ylim))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True)

    # Save the plot
    output_path = f"{args.data_dir}/transformer_training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")


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
