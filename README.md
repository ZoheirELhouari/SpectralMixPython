# Data augmentation techniques applied to SpectralMix Algorithm

Based on the work of Ylli Sadikaj, Yllka Velaj, Sahar Behzadi, Claudia Plant
Paper Link: https://arxiv.org/pdf/2311.01840

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

```
pip install -r requirements.txt
```

## Usage

template to run the project:
```
python main.py [datasetName] [augmentation method]
```

Run SpectralMix WITH data augmentation : 
```
python main.py acm graph_sampling
```

Run SpectralMix WITHOUT data augmentation:
```
python main.py acm no_augmentation
```
data augmentation techniques to be tested: 
```
Feature Masking: feature_masking 
Feature Shuffling: feature_shuffling
Feature Propagation: feature_propagation (Implemented by can’t be applied to the current datasets) 
Edge Perturbation: edge_perturbation
Graph Diffusion (Heat): graph_diffusion_heat
Graph Diffusion (PPR): graph_diffusion_ppr
Graph Rewiring: graph_rewiring
Graph sampling: graph_sampling
Node Dropping: node_dropping
Node Insertion: node_insertion
```

## Results: 
After successfully running running the project, the results will be added as a new row inside the csv file in the project folder called: ```experiment_results.csv```

## Contributing

Guidelines for contributing to your project.

## License

Information about the license for your project.
