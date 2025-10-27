"""Rebuild trained models from cello samples - embracingearth.space"""

from pathlib import Path
from ddsp_sample_based import SampleBasedDDSP
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebuild_models():
    """Rebuild cello_samples.pkl from training data"""
    
    # Initialize sample-based DDSP
    sample_based = SampleBasedDDSP()
    
    # Load samples from training directory
    samples_dir = Path("neural_cello_training_samples/filtered_20250808_010724_batch")
    
    if not samples_dir.exists():
        logger.error(f"Training samples not found at {samples_dir}")
        return False
    
    logger.info(f"Loading samples from {samples_dir}")
    sample_based.load_samples(samples_dir)
    
    if not sample_based.is_loaded:
        logger.error("Failed to load any samples!")
        return False
    
    logger.info(f"Loaded {len(sample_based.samples_library)} pitch ranges with samples")
    
    # Save the trained model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "cello_samples.pkl"
    
    logger.info(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'samples_library': sample_based.samples_library,
            'sample_rate': sample_based.sample_rate
        }, f)
    
    logger.info(f"Model saved successfully! Total pitches: {len(sample_based.samples_library)}")
    
    return True

if __name__ == "__main__":
    rebuild_models()

