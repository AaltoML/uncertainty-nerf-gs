from dataclasses import dataclass
from pathlib import Path

from typing import Optional, Literal, List, Union


@dataclass
class EvalUncertainty:
    load_config: Path
    # Path to the trained model configuration.

    dataset_path: Path
    # Path to the dataset we are testing 

    output_path: Path = Path("output.json")
    # Path to save the output metrics.

    render_output_path: Optional[Path] = None
    # Path to save the rendered images.

    save_all_ause: bool = False
    # Save AUSE metrics for all test images.

    seed: int = 42
    # Random seed to make have reproducible resu

    eval_depth: bool = True
    # eval uncertainty for depth

    eval_rgb: bool = True
    # eval rgb uncertainty

    plot_ause: bool = False
    # Plot and save AUSE metrics for all test images.

    save_rendered_images: bool = False
    # Plot and save all rendered images or not.  
    
    min_rgb_std_for_nll: float = 3e-2
    # minimum standard deviation when computing the NLL metric for RGB aka the "nugget" term.  
    
    min_depth_std_for_nll: float = 2.0
    # minimum standard deviation when computing the NLL metric for Depth.

    unc_max: float = 1.0
    # Maximum value for uncertainty with clipping. Useful for visualization.
    
    unc_min: float = 0.0
    # Minimum value for uncertainty with clipping.    


@dataclass
class LaplaceConfig(EvalUncertainty):
    prior_precision: float = 1.0
    # Prior precision
    
    n_samples: int = 100
    # samples for the Laplace MC sapling 

    n_iters: int = 300
    # number of iterations to compute the diagonal of the GGN matrix 

    use_deterministic_density: bool = False 
    # use deterministic or sampled density values 


@dataclass
class EnsembleConfig(EvalUncertainty):
    load_config: List[Path]
    # List of paths to the trained model configurations. Must be a list of at
    # least two paths.
    

@dataclass
class MCDropoutConfig(EvalUncertainty):
    mc_samples: Optional[int] = None
    # Number of samples to use for Monte Carlo dropout.

@dataclass
class ActiveNerfactoConfig(EvalUncertainty):
    eval_depth: bool = True
    # eval uncertainty for depth cannot be evaluated for ActiveNerfacto
    
@dataclass
class ActiveSplatfactoConfig(EvalUncertainty):
    eval_depth: bool = False
    # eval uncertainty for depth cannot be evaluated for ActiveNerfacto

@dataclass
class RobustNerfactoConfig(EvalUncertainty):
    eval_depth: bool = False
    # eval uncertainty for depth cannot be evaluated for RobustNerfacto

    eval_rgb: bool = False
    # eval rgb uncertainty cannot be evaluated for RobustNerfacto

# Union of all configuration classes for evaluation.
EvalConfigs = Union[
    LaplaceConfig, 
    EnsembleConfig,
    MCDropoutConfig,
    ActiveNerfactoConfig,
    ActiveSplatfactoConfig,
    RobustNerfactoConfig
]