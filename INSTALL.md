This repo is based on [detectron2](https://github.com/facebookresearch/detectron2) framework.

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, optional, needed by demo and visualization


### Build the source code

After having the above dependencies and gcc & g++ ≥ 5, run:
```
# install it from a local clone:
git clone https://github.com/dragonlee258079/Saliency-Ranking.git
cd Saliency-Ranking && python -m pip install -e .

# Or if you are on macOS
# CC=clang CXX=clang++ python -m pip install -e .
```

## NOTE
To __rebuild__ detectron2 that's built from a local clone, use `rm -rf build/ **/*.so` to clean the
old build first. You often need to rebuild detectron2 after reinstalling PyTorch.
