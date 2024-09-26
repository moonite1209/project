import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from mydataset import Autoencoder_dataset
from model import Autoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-s', type=str, required=True)
    parser.add_argument('--segment_folder', type=str, default='semantic')
    parser.add_argument('--ckpt_folder', type=str, default='autoencoder')
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )
    args = parser.parse_args()
    
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    ckpt_path = os.path.join(dataset_path, args.ckpt_folder, 'best_ckpt.pth')

    data_dir = os.path.join(dataset_path, args.segment_folder)
    output_dir = os.path.join(dataset_path, args.segment_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = torch.load(ckpt_path)
    train_dataset = Autoencoder_dataset(data_dir)

    test_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=256,
        shuffle=False, 
        num_workers=16, 
        drop_last=False   
    )


    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    features=[]
    for idx, feature in tqdm(enumerate(test_loader)):
        data = feature.to("cuda:0")
        with torch.no_grad():
            outputs = model.encode(data)
        features.append(outputs)

    os.makedirs(output_dir, exist_ok=True)
    np.save(f'{output_dir}/semantics.npy', torch.cat(features).detach().cpu().numpy())