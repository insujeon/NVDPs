import os, argparse, time, random, uuid
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from models import *
from utils import *

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

######################### Parameters and Device Setting #########################
parser = argparse.ArgumentParser(description='main_2d.py')
parser.add_argument('--exp', default='2D_exp', help='experiment name')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', default='celeba', type=str, choices=['mnist', 'celeba', 'omni'])
parser.add_argument('--model', default='NVDP', type=str, choices=['NVDP', 'NVDP+det', 'CNP', 'NP', 'NP+det', 'NP+rev', 'NP+det+rev'], help='model type')
parser.add_argument('--h_dim', type=int, default=128, help='overall NNs hidden dimension')
parser.add_argument('--r_dim', type=int, default=128, help='deterministic path vectors dimension')
parser.add_argument('--z_dim', type=int, default=128, help='latent path vectors dimension')
parser.add_argument('--lr', type=float, default=4e-4, help='learning rate')
parser.add_argument('--beta', type=float, default=0.5, help='KL divergence hyperparameter') 
parser.add_argument('--alpha', type=float, default=-2)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--sample_size', type=int, default=3)
parser.add_argument('--epochs', type=int, default=500)
args = parser.parse_args()

start_time = time.time()
args.cuda = torch.cuda.is_available()
args.device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#np.random.seed(args.seed)
random.seed(args.seed)

from data.data_independent import *

######################### Dataset and Model ####################################

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

if args.dataset =='mnist':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    args.imgdim = 28
    args.x_dim = 2
    args.y_dim = 1
    args.num_point = 784

elif args.dataset == 'omni':
    train_loader = DataLoader(dataset=OmniDataset(train=True), batch_size=args.batch_size, shuffle=True, drop_last=True,**kwargs)
    #test_loader = DataLoader(dataset=OmniDataset(train=False), batch_size=args.batch_size, shuffle=True, drop_last=True,**kwargs)
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    args.imgdim = 28
    args.x_dim = 2
    args.y_dim = 1
    args.num_point = 784

elif args.dataset == 'celeba':
    transform = transforms.Compose([
        #transforms.CenterCrop(crop),
        #transforms.Resize(32),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    celeba_data = CelebADataset('./data/CelebA_cropped64_new/cropped_32', transform=transform)
    train_loader = DataLoader(celeba_data, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)    
    test_loader = DataLoader(celeba_data, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)    
    args.imgdim = 32
    args.x_dim = 2
    args.y_dim = 3
    args.num_point = 1024

args.x_grid = generate_grid2(args.imgdim, args.imgdim, args.device)

if args.model == 'NVDP':
    args.det_path = False
    args.rev = True
    model = ModelNVDP(args).to(args.device)

elif args.model == 'NVDP+det':
    args.det_path = True
    args.rev = True
    model = ModelNVDP(args).to(args.device)

if args.model == 'CNP':
    args.det_path = True
    args.latent_path = False
    args.rev = False
    model = LatentModel(args).to(args.device)

elif args.model == 'NP':
    args.det_path = False
    args.latent_path = True
    args.rev = False
    model = LatentModel(args).to(args.device)

elif args.model == 'NP+det':
    args.det_path = True
    args.latent_path = True
    args.rev = False
    model = LatentModel(args).to(args.device)

elif args.model == 'NP+rev':
    args.det_path = False
    args.latent_path = True
    args.rev = True
    model = LatentModel(args).to(args.device)

elif args.model == 'NP+det+rev':
    args.det_path = True
    args.latent_path = True
    args.rev = True
    model = LatentModel(args).to(args.device)

optimizer = optim.Adam(model.parameters(), betas=(0.99, 0.999), lr=args.lr)


######################### Experiment Logging Information ############################
args.uuid = str(uuid.uuid1())[:8]
args.exp = args.exp + f'_{args.model}_{args.dataset}_(det:{args.det_path},a:{args.alpha},b:{args.beta},lr:{args.lr})' + f'_{args.uuid}'

args.event_dir = f'./runs/{args.dataset}/events' + '/' + args.exp
args.img_dir = f'./runs/{args.dataset}/imgs' + '/' + args.exp
args.ckpt_dir = f'./runs/{args.dataset}/ckpt' + '/' + args.exp
if not os.path.exists(args.img_dir): os.makedirs(args.img_dir)
if not os.path.exists(args.event_dir): os.makedirs(args.event_dir)
if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)

writer = SummaryWriter(args.event_dir)

args.log_interval = 100
#args.test_interval = sorted(set( list(range(0,args.epochs+1,100)) ))

#g_iter = 0
train_loss = [] 
train_LL = []
train_KLD = []
train_mse_loss = []
train_Recon_LL = []
train_Pred_LL = []
train_g_iter = []
test_LL = []
test_Recon_LL = []
test_Pred_LL = []
test_g_iter = []

print("Experiment Setup Done")


############################ Train and Test routine #################################
def train(epoch):
    model.train()
    LL2 = 0 
    KLD2 = 0
    mse_loss = 0
    Recon_LL = 0
    Pred_LL = 0

    for batch_idx, (y_t, _) in enumerate(train_loader):
        model.apply(renorm_weights)
        optimizer.zero_grad()

        x_t = args.x_grid.expand(args.batch_size, -1, -1)
        y_t = y_t.to(args.device).view(args.batch_size, -1, args.y_dim)

        ################## ANP setting ##################
        num_context = random.randint(3, 196)  # number of context points ([3,197))
        num_target = random.randint(num_context+1, 200) # number of target points [m+1, 200]
        rand_idx = get_context_idx(num_target, args.device, num_point=args.num_point)
        x_c = idx_to_x(rand_idx[:num_context], args.batch_size, args.x_grid)
        y_c = idx_to_y(rand_idx[:num_context], y_t)
        x_t = idx_to_x(rand_idx, args.batch_size, args.x_grid)
        y_t = idx_to_y(rand_idx, y_t)
        ################################################

        LL_temp = []
        if args.model in {'NVDP', 'NVDP+det'}:
            for sample in range(args.sample_size):
                y_pred = model(x_c, y_c, x_t, y_t)
                y_dist = Normal(y_pred[0], y_pred[1])
                log_p = y_dist.log_prob(y_t-0.5)
                LL_temp.append(log_p.mean())
            r_t = model.det_encoder2(x_t, y_t, x_t)
            KLD = kl_div2(model, r_t)

        elif args.model in {'NP', 'NP+det', 'CNP'}:
            for sample in range(args.sample_size):
                y_pred, z_t, z_c = model(x_c, y_c, x_t, y_t)
                y_dist = Normal(y_pred[0], y_pred[1])
                log_p = y_dist.log_prob(y_t-0.5)
                LL_temp.append(log_p.mean())
            KLD = kl_div_gaussians3(z_t[0], z_t[1], z_c[0], z_c[1])

        elif args.model in {'NP+rev', 'NP+det+rev'}:

            for sample in range(args.sample_size):
                y_pred, z_t, z_c = model(x_c, y_c, x_t, y_t)
                y_dist = Normal(y_pred[0], y_pred[1])
                log_p = y_dist.log_prob(y_t-0.5)
                LL_temp.append(log_p.mean())
            KLD = kl_div_gaussians3(z_c[0], z_c[1], z_t[0], z_t[1]) 

        LL = torch.stack(LL_temp).mean()
        loss = -LL + args.beta * KLD / (x_t.size(1))
        loss.backward()
        optimizer.step()

        LL2 += LL.item()
        KLD2 += KLD.item()
        mse_loss += F.mse_loss(y_pred[0],y_t-0.5).mean().item()
        Recon_LL += log_p[:, :num_context, :].mean().item()
        Pred_LL += log_p[:, num_context:, :].mean().item()

        if batch_idx % args.log_interval == 0:
            print(f'(train) Epoch:{epoch}, Batch:{batch_idx}, loss: {loss.item():.4f}')

    print(args.exp + f' (training) Epoch: {epoch}, Average loss: {loss.item():.4f}')
    writer.add_scalar('train/LL', LL2/(batch_idx+1), epoch)
    writer.add_scalar('train/KLD', KLD2/(batch_idx+1), epoch)
    writer.add_scalar('train/mse_loss', mse_loss/(batch_idx+1), epoch)
    writer.add_scalar('train/Recon_LL', Recon_LL/(batch_idx+1), epoch)
    writer.add_scalar('train/Pred_LL', Pred_LL/(batch_idx+1), epoch)
    train_LL.append(LL2/(batch_idx+1))
    train_KLD.append(KLD2/(batch_idx+1))
    train_mse_loss.append(mse_loss/(batch_idx+1))
    train_Recon_LL.append(Recon_LL/(batch_idx+1))
    train_Pred_LL.append(Pred_LL/(batch_idx+1))
    train_g_iter.append(epoch)
    print(f'====> Train LL: {LL2/(batch_idx+1):.4f}')


def test(epoch):
    model.eval()

    LL = []
    Recon_LL = []
    Pred_LL = []

    with torch.no_grad():
        for batch_idx, (y_t,_) in enumerate(test_loader):

            x_t = args.x_grid.expand(args.batch_size, -1, -1)
            y_t = y_t.to(args.device).view(args.batch_size, -1, args.y_dim)

            if batch_idx == 0:  # save PNG of reconstructed examples
                plot_Ns = [10, 30, 100, args.num_point]
                num_examples = min(args.batch_size, 16)
                for N in plot_Ns:
                    recons = []
                    context_idx = get_context_idx(N, args.device, num_point=args.num_point)
                    x_c = idx_to_x(context_idx, args.batch_size, args.x_grid)
                    y_c = idx_to_y(context_idx, y_t)

                    for d in range(5):
                        if args.model in {'NVDP', 'NVDP+det'}:
                            y_hat = model(x_c, y_c, x_t)
                        elif args.model in {'NP', 'NP+det', 'CNP', 'NP+rev', 'NP+det+rev'}:
                            y_hat, _, _ = model(x_c, y_c, x_t)

                        #recons.append(inv_normalize(y_hat[0][:num_examples]).clamp(min=0.0,max=1.0))
                        recons.append( torch.clamp(y_hat[0][:num_examples]+0.5,0,1) )

                    recons = torch.cat(recons).view(-1, args.y_dim, args.imgdim, args.imgdim).expand(-1, 3, -1, -1)
                    background = torch.tensor([0., 0., 0.], device=args.device)
                    background = background.view(1, -1, 1).expand(num_examples, 3, args.num_point).contiguous()
                    context_pixels = y_t[:num_examples].view(num_examples, args.y_dim, -1)[:, :, context_idx]
                    context_pixels = context_pixels.expand(num_examples, 3, -1)
                    background[:, :, context_idx] = context_pixels
                    comparison = torch.cat([background.view(-1, 3, args.imgdim, args.imgdim), recons])
                    #save_image(comparison.cpu(), f'results/ep_{epoch}_cps_{N}.png', nrow=num_examples)
                    #save_image(comparison.cpu(), args.img_dir + f'/ep_{epoch}_cps_{N}.png', nrow=num_examples)

                    fig = plt.figure()
                    plt.title(f"Conditioned on {N} context points (epoch {epoch}).")
                    grid_img = make_grid(comparison, nrow=num_examples)
                    writer.add_image(f'cps_{N}', grid_img, epoch)


            ################## ANP setting ##################
            num_context = random.randint(3, 196)  # number of context points ([3,197))
            num_target = args.num_point # number of target points 784
            rand_idx = get_context_idx(num_target, args.device, num_point=args.num_point)
            x_c = idx_to_x(rand_idx[:num_context], args.batch_size, args.x_grid)
            y_c = idx_to_y(rand_idx[:num_context], y_t)
            x_t = idx_to_x(rand_idx, args.batch_size, args.x_grid)
            y_t = idx_to_y(rand_idx, y_t)
            ################################################

            if args.model in {'NVDP', 'NVDP+det'}:
                # concat = x_t
                # if args.det_path:
                #     r_d = model.det_encoder(x_c, y_c, x_t)
                #     concat = torch.cat([concat, r_d.repeat(1, x_t.size(1), 1)], dim=-1)
                # r = model.det_encoder2(x_c, y_c, x_t)
                # y_pred = model.decoder(concat, r)
                # y_dist = Normal(y_pred[0], y_pred[1]) 
                # #log_p = y_dist.log_prob(y_t)

                y_pred = model(x_c, y_c, x_t, y_t)
                y_dist = Normal(y_pred[0], y_pred[1])
                log_p = y_dist.log_prob(y_t-0.5)

            elif args.model in {'NP', 'NP+det', 'CNP', 'NP+rev', 'NP+det+rev'}:
                # r = None
                # if args.det_path:
                #     r = model.det_encoder(x_c, y_c, x_t)
                #     #r = r.unsqueeze(1).expand(-1, x_t.size(1), -1)
                # z_c = model.latent_encoder(x_c, y_c)
                # z = z_c[0].unsqueeze(1).expand(-1, x_t.size(1), -1) # mean of z_c for validation
                # y_pred = model.decoder(z, x_t, r)
                # y_dist = Normal(y_pred[0], y_pred[1]) 
                # log_p = y_dist.log_prob(y_t-0.5)

                y_pred, _, _ = model(x_c, y_c, x_t, y_t)
                y_dist = Normal(y_pred[0], y_pred[1])
                log_p = y_dist.log_prob(y_t-0.5)

            #LL += log_p.mean().item() # LL for all given context
            LL.append(log_p.mean().item())
            Recon_LL.append(log_p[:, :num_context, :].mean().item())
            Pred_LL.append(log_p[:, num_context:, :].mean().item())
            
    writer.add_scalar(f'test/LL', torch.Tensor(LL).mean().item(), epoch)
    writer.add_scalar(f'test/Recon_LL', torch.Tensor(Recon_LL).mean().item(), epoch)
    writer.add_scalar(f'test/Pred_LL', torch.Tensor(Pred_LL).mean().item(), epoch)
    writer.add_scalar(f'test/LL(STD)', torch.Tensor(LL).std().item(), epoch)
    writer.add_scalar(f'test/Recon_LL(STD)', torch.Tensor(Recon_LL).std().item(), epoch)
    writer.add_scalar(f'test/Pred_LL(STD)', torch.Tensor(Pred_LL).std().item(), epoch)

    test_LL.append(torch.Tensor(LL).mean().item())
    test_Recon_LL.append(torch.Tensor(Recon_LL).mean().item())
    test_Pred_LL.append(torch.Tensor(Pred_LL).mean().item())
    test_g_iter.append(epoch)

    print(f'====> Test LL: {torch.Tensor(LL).mean().item():.4f}')


######################################## main ########################################
print("Training Start:" + args.exp)
for epoch in range(1, args.epochs +1): 
    train(epoch)
    writer.flush()
    test(epoch)
    writer.flush()

    # if epoch in args.test_interval:
    torch.save(
            {'model_state': model.state_dict(), 
                'args': args, 

                'train_loss'       :train_loss,
                'train_LL'        :train_LL,
                'train_Recon_LL'   :train_Recon_LL,
                'train_Pred_LL'     :train_Pred_LL,
                'train_KLD'        :train_KLD,
                'train_mse_loss'   :train_mse_loss,
                'train_g_iter'     :train_g_iter,

                'test_LL'         :test_LL,
                'test_Recon_LL'   :test_Recon_LL,
                'test_Pred_LL'     :test_Pred_LL,
                'test_g_iter'      :test_g_iter,

                }, args.ckpt_dir+'/ckpt.pt')
        # test(epoch, n_test=30)
    # writer.flush()

####################################### Final ###############################
print("End Experiment" + args.exp)
writer.flush()
writer.close()
args.total_time = (time.time() - start_time)/60
print(f"--- {args.total_time: 4f} minutes ---")
