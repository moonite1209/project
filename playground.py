import os
from scene import Scene, GaussianModel

if __name__ == '__main__':
    dataset='fern_8'
    dataset_path=os.path.join('./data',dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
