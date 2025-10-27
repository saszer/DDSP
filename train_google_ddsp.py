"""Train Google DDSP model on cello samples - embracingearth.space"""

from pathlib import Path
from ddsp_trainer import DDSPTrainer, TrainingConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_google_ddsp():
    """Train Google DDSP model from cello samples"""
    
    # Configuration
    config = TrainingConfig(
        sample_rate=48000,
        n_fft=2048,
        hop_length=256,
        n_harmonics=60,
        n_noise_frames=65,
        batch_size=32,
        n_epochs=50,
        learning_rate=1e-4
    )
    
    # Initialize trainer
    trainer = DDSPTrainer(config)
    
    # Train the model (pass directory path)
    samples_dir = Path("neural_cello_training_samples/filtered_20250808_010724_batch")
    
    if not samples_dir.exists():
        logger.error(f"Training samples not found at {samples_dir}")
        return False
    
    logger.info("Training Google DDSP model...")
    trainer.train(samples_dir)
    
    # Save the trained model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "cello_ddsp_model.pkl"
    
    logger.info(f"Saving trained model to {model_path}")
    trainer.save_model(model_path)
    
    logger.info("Google DDSP model trained and saved successfully!")
    
    return True

if __name__ == "__main__":
    train_google_ddsp()

