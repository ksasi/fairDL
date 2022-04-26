import torch
import sys
import os
import argparse
import pickle
import PIL.Image
import numpy as np

# Ref : https://github.com/NVlabs/stylegan2-ada-pytorch

parser = argparse.ArgumentParser(description='StyleGAN2 generator to generate face images')

parser.add_argument("--num", default=50, type=int, help='Number of images to be generated(default value is 2000)')
parser.add_argument("--outdir", default="/workspace/fairDL/data/synthface", type=str, help='path to save the generated images(default is ../data/stylegan2/)')


def main():
    global args
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    try:
        sys.path.index('/workspace/stylegan2-ada-pytorch')
    except:
        sys.path.append('/workspace/stylegan2-ada-pytorch')

    #print(sys.path)
    os.makedirs(args.outdir, exist_ok=True)

    with open('/workspace/fairDL/src/ffhq.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()

    #directions = np.load('/workspace/fairDL/data/latents/glasses.npy')
    #print(directions.shape, flush=True)
    #print(len(directions), flush=True)

    for img_num in range(args.num):
        z_rand = torch.randn([18, G.z_dim]).cuda()
        #z_direction = torch.from_numpy(directions[10]).unsqueeze(0).cuda()
        #z = z_rand + 3*z_direction
        z = z_rand
        print(z.shape)
        print(G.z_dim, flush=True)
        c = None
        smile_direction = np.load('/workspace/stylegan-encoder/ffhq_dataset/latent_directions/smile.npy')

        #img = G(z, c, truncation_psi=1, noise_mode='const')
        os.makedirs(f'{args.outdir}/id_{img_num}', exist_ok=True)
        new_z = z.clone().cpu().numpy()
        ####print("HERE")
        ####print(new_z.shape)
        ####print(smile_direction.shape)
        new_z[:8] = (z.cpu().numpy() - 5*smile_direction)[:8]
        z_final = torch.from_numpy(new_z).cuda()
        img1 = G(z_final, c, truncation_psi=1, noise_mode='random')
        img1 = (img1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        #print(img[0])
        #print("\n")
        #print(img[0].cpu().numpy())
        PIL.Image.fromarray(img1[0].cpu().numpy(), 'RGB').save(f'{args.outdir}/id_{img_num}/{img_num}_1.png')

        new_z = z.clone().cpu().numpy()
        new_z[:8] = (z.cpu().numpy() + 0*smile_direction)[:8]
        z_final = torch.from_numpy(new_z).cuda()
        img2 = G(z_final, c, truncation_psi=1, noise_mode='random')
        img2 = (img2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img2[0].cpu().numpy(), 'RGB').save(f'{args.outdir}/id_{img_num}/{img_num}_2.png')

        new_z = z.clone().cpu().numpy()
        new_z[:8] = (z.cpu().numpy() + 2*smile_direction)[:8]
        z_final = torch.from_numpy(new_z).cuda()
        img3 = G(z_final, c, truncation_psi=1, noise_mode='const')
        img3 = (img3.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img3[0].cpu().numpy(), 'RGB').save(f'{args.outdir}/id_{img_num}/{img_num}_3.png')

        new_z = z.clone().cpu().numpy()
        new_z[:8] = (z.cpu().numpy() + 4*smile_direction)[:8]
        z_final = torch.from_numpy(new_z).cuda()
        img4 = G(z_final, c, truncation_psi=1, noise_mode='const')
        img4 = (img4.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img4[0].cpu().numpy(), 'RGB').save(f'{args.outdir}/id_{img_num}/{img_num}_4.png')

if __name__ == "__main__":
    main()

    

