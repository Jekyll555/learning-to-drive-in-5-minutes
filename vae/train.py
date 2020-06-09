"""
Train a VAE model using saved images in a folder
"""
import argparse
import os

import cv2
import numpy as np
from stable_baselines.common import set_global_seeds
from tqdm import tqdm

from config import ROI
from vae.controller import VAEController
from .data_loader import DataLoader
from .model import ConvVAE

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folders', help='Path to folders containing images for training', type=str,
                    nargs='+', required=True)
parser.add_argument('--z-size', help='Latent space', type=int, default=512)
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('--n-samples', help='Max number of samples', type=int, default=-1)
parser.add_argument('--batch-size', help='Batch size', type=int, default=64)
parser.add_argument('--learning-rate', help='Learning rate', type=float, default=1e-4)
parser.add_argument('--kl-tolerance', help='KL tolerance (to cap KL loss)', type=float, default=0.5)
parser.add_argument('--beta', help='Weight for kl loss', type=float, default=1.0)
parser.add_argument('--n-epochs', help='Number of epochs', type=int, default=10)
parser.add_argument('--verbose', help='Verbosity', type=int, default=1)
args = parser.parse_args()

set_global_seeds(args.seed)

folders, images = [], []
for folder in args.folders:
    if not folder.endswith('/'):
        folder += '/'
    folders.append(folder)
    images_ = [folder + im for im in os.listdir(folder) if im.endswith('.jpg')]
    print("{}: {} images".format(folder, len(images_)))
    images.append(images_)

vae = ConvVAE(z_size=args.z_size,
              batch_size=args.batch_size,
              learning_rate=args.learning_rate,
              kl_tolerance=args.kl_tolerance,
              beta=args.beta,
              is_training=True,
              reuse=False)


images = np.concatenate(images)
n_samples = len(images)

if args.n_samples > 0:
    n_samples = min(n_samples, args.n_samples)

print("{} images".format(n_samples))

# indices for all time steps where the episode continues
indices = np.arange(n_samples, dtype='int64')
np.random.shuffle(indices)

# split indices into minibatches. minibatchlist is a list of lists; each
# list is the id of the observation preserved through the training
minibatchlist = [np.array(sorted(indices[start_idx:start_idx + args.batch_size]))
                 for start_idx in range(0, len(indices) - args.batch_size + 1, args.batch_size)]

data_loader = DataLoader(minibatchlist, images, n_workers=2)

vae_controller = VAEController(z_size=args.z_size)
vae_controller.vae = vae
best_loss = np.inf
vae_id = int(time.time())
save_path = "logs/vae-{}_{}.pkl".format(args.z_size, vae_id)
best_model_path = "logs/vae-{}_{}_best.pkl".format(args.z_size, vae_id)
os.makedirs(os.path.dirname(save_path), exist_ok=True)

for epoch in range(args.n_epochs):
    pbar = tqdm(total=len(minibatchlist))
    for obs, target_obs in data_loader:
        feed = {vae.input_tensor: obs, vae.target_tensor: target_obs}
        (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
            vae.loss,
            vae.r_loss,
            vae.kl_loss,
            vae.global_step,
            vae.train_op
        ], feed)
        pbar.update(1)
    pbar.close()
    print("Epoch {:3}/{}".format(epoch + 1, args.n_epochs))
    print("VAE: optimization step", (train_step + 1), train_loss, r_loss, kl_loss)

    # Update params
    vae_controller.set_target_params()
    # Load test image
    if args.verbose >= 1:
        image_idx = np.random.randint(n_samples)
        image = cv2.imread(images[image_idx])
        r = ROI
        im = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        encoded = vae_controller.encode(im)
        reconstructed_image = vae_controller.decode(encoded)[0]
        # Plot reconstruction
        cv2.imshow("Original", image)
        cv2.imshow("Reconstruction", reconstructed_image)
        cv2.waitKey(1)

save_path = "logs/vae-{}".format(args.z_size)
os.makedirs(save_path, exist_ok=True)
print("Saving to {}".format(save_path))
vae_controller.set_target_params()
vae_controller.save(save_path)
