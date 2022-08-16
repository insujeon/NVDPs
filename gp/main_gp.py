import os, argparse, time, random, uuid
import torch
from torch import optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
# from torchvision.utils import save_image, make_grid

from models import *
from data.gp_curves_gpu import GPCurvesReader
from utils import *

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

######################### Parameters and Device Setting #########################
parser = argparse.ArgumentParser(description='main_gp.py')
parser.add_argument('--exp', default='GP_exp_bias', help='experiment name')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', default='gp', type=str)
parser.add_argument('--model', default='NVDP', type=str, choices=['NVDP', 'NVDP+det', 'CNP', 'NP', 'NP+det', 'NP+rev', 'NP+det+rev'], help='model type')
parser.add_argument('--h_dim', type=int, default=128, help='overall NNs hidden dimension')
parser.add_argument('--r_dim', type=int, default=128, help='deterministic path vectors dimension')
parser.add_argument('--z_dim', type=int, default=128, help='latent path vectors dimension')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--beta', type=float, default=1.4, help='KL divergence hyperparameter') 
parser.add_argument('--alpha', type=float, default=-2)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--sample_size', type=int, default=3)
parser.add_argument('--epochs', type=int, default=500000)
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


######################### Dataset and Model ####################################
if args.dataset == 'gp':
    dataset_train = GPCurvesReader(batch_size=16, max_num_context=96, random_kernel_parameters=True)
    dataset_test = GPCurvesReader(batch_size=1, max_num_context=96, testing=True, random_kernel_parameters=True)
    args.x_dim = 1
    args.y_dim = 1
#else args.dataset = 'mnist':

if args.model == 'NVDP':
    args.det_path = False
    args.rev = True
    #args.det_cross_attn_type = 'uniform'
    model = ModelNVDP(args).to(args.device)

elif args.model == 'NVDP+det':
    args.det_path = True
    args.rev = True
    #args.det_cross_attn_type = 'uniform'
    model = ModelNVDP(args).to(args.device)

if args.model == 'CNP':
    args.det_path = True
    args.latent_path = False
    args.rev = False
    #args.det_cross_attn_type = 'uniform'
    model = LatentModel(args).to(args.device)

elif args.model == 'NP':
    args.det_path = False
    args.latent_path = True
    args.rev = False
    #args.det_cross_attn_type = 'uniform'
    model = LatentModel(args).to(args.device)

elif args.model == 'NP+det':
    args.det_path = True
    args.latent_path = True
    args.rev = False
    #args.det_cross_attn_type = 'uniform'
    model = LatentModel(args).to(args.device)

elif args.model == 'NP+rev':
    args.det_path = False
    args.latent_path = True
    args.rev = True
    #args.det_cross_attn_type = 'uniform'
    model = LatentModel(args).to(args.device)

elif args.model == 'NP+det+rev':
    args.det_path = True
    args.latent_path = True
    args.rev = True
    #args.det_cross_attn_type = 'uniform'
    model = LatentModel(args).to(args.device)

optimizer = optim.Adam(model.parameters(), betas=(0.99, 0.999), lr=args.lr)


######################### Experiment Logging Information ############################
args.uuid = str(uuid.uuid1())[:8]
args.exp = args.exp + f'_{args.model}(det:{args.det_path},a:{args.alpha},b:{args.beta},lr:{args.lr})' + f'_{args.uuid}'

args.event_dir = f'./runs/{args.dataset}/events' + '/' + args.exp
args.img_dir = f'./runs/{args.dataset}/imgs' + '/' + args.exp
args.ckpt_dir = f'./runs/{args.dataset}/ckpt' + '/' + args.exp
if not os.path.exists(args.img_dir): os.makedirs(args.img_dir)
if not os.path.exists(args.event_dir): os.makedirs(args.event_dir)
if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)

writer = SummaryWriter(args.event_dir)

args.log_interval = 100
args.test_interval = sorted(set( list(range(0,args.epochs+1,100)) ))

# g_iter = 0
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
    model.apply(renorm_weights)
    optimizer.zero_grad()

    data_train = dataset_train.generate_curves()
    (x_c, y_c), x_t = data_train.query
    y_t = data_train.target_y

    LL_temp = []
    if args.model in {'NVDP', 'NVDP+det'}:
        for sample in range(args.sample_size):
            y_pred = model(x_c, y_c, x_t, y_t)
            y_dist = Normal(y_pred[0], y_pred[1])
            log_p = y_dist.log_prob(y_t)
            LL_temp.append(log_p.mean())
        r_t = model.det_encoder2(x_t, y_t, x_t)
        KLD = kl_div2(model, r_t)

    elif args.model in {'NP', 'NP+det', 'CNP'}:
        for sample in range(args.sample_size):
            y_pred, z_t, z_c = model(x_c, y_c, x_t, y_t)
            y_dist = Normal(y_pred[0], y_pred[1])
            log_p = y_dist.log_prob(y_t)
            LL_temp.append(log_p.mean())
        KLD = kl_div_gaussians3(z_t[0], z_t[1], z_c[0], z_c[1])

    elif args.model in {'NP+rev', 'NP+det+rev'}:

        for sample in range(args.sample_size):
            y_pred, z_t, z_c = model(x_c, y_c, x_t, y_t)
            y_dist = Normal(y_pred[0], y_pred[1])
            log_p = y_dist.log_prob(y_t)
            LL_temp.append(log_p.mean())
        KLD = kl_div_gaussians3(z_c[0], z_c[1], z_t[0], z_t[1]) 


    LL = torch.stack(LL_temp).mean()
    loss = -LL + args.beta * KLD / (x_t.size(1))
    loss.backward()
    optimizer.step()

    if epoch % args.log_interval == 0:
        print(args.exp + f' (training) Epoch: {epoch}, Average loss: {loss.item():.4f}')
        writer.add_scalar('train/LL', LL.cpu().item(), epoch)
        writer.add_scalar('train/KLD', KLD.cpu().item(), epoch)
        writer.add_scalar('train/loss', loss.cpu().item(), epoch)
        mse_loss = F.mse_loss(y_pred[0], y_t).mean().cpu().item()
        Recon_LL = log_p[:, :y_c.size(1), :].mean().cpu().item()
        Pred_LL = log_p[:, y_c.size(1):, :].mean().cpu().item()
        writer.add_scalar('train/mse_loss', mse_loss, epoch)
        writer.add_scalar('train/Recon_LL', Recon_LL, epoch)
        writer.add_scalar('train/Pred_LL', Pred_LL, epoch)
        train_LL.append(LL.cpu().item())
        train_KLD.append(KLD.cpu().item())
        train_loss.append(loss.cpu().item())
        train_mse_loss.append(mse_loss)
        train_Recon_LL.append(Recon_LL)
        train_Pred_LL.append(Pred_LL)
        train_g_iter.append(epoch)


def test(epoch, n_test=30, title='test', fixed_img=False):
    model.eval()

    LL = []
    Recon_LL = []
    Pred_LL = []

    with torch.no_grad():
        for i in range(n_test):
            data_test = dataset_test.generate_curves()
            (x_c, y_c), x_t = data_test.query
            y_t = data_test.target_y
            x_t = torch.cat([x_t, x_c], dim=1)
            y_t = torch.cat([y_t, y_c], dim=1)

            if args.model in {'NVDP', 'NVDP+det'}:
                concat = x_t
                if args.det_path:
                    r_d = model.det_encoder(x_c, y_c, x_t)
                    concat = torch.cat([concat, r_d.repeat(1, x_t.size(1), 1)], dim=-1)
                r = model.det_encoder2(x_c, y_c, x_t)
                y_pred = model.decoder(concat, r)
                y_dist = Normal(y_pred[0], y_pred[1]) 
                log_p = y_dist.log_prob(y_t)

            elif args.model in {'NP', 'NP+det', 'CNP', 'NP+rev', 'NP+det+rev'}:
                r = None
                if args.det_path:
                    r = model.det_encoder(x_c, y_c, x_t)
                    #r = r.unsqueeze(1).expand(-1, x_t.size(1), -1)
                z_c = model.latent_encoder(x_c, y_c)
                z = z_c[0].unsqueeze(1).expand(-1, x_t.size(1), -1) # mean of z_c for validation
                y_pred = model.decoder(z, x_t, r)
                y_dist = Normal(y_pred[0], y_pred[1]) 
                log_p = y_dist.log_prob(y_t)

            # LL, Recon LL, Predictive LL
            LL.append(log_p.mean().cpu().item()) # LL for all given context
            Recon_LL.append(log_p[:, -x_c.size(1):, :].mean().cpu().item()) # LL for only context given context
            Pred_LL.append(log_p[:, :-x_c.size(1), :].mean().cpu().item()) # LL for only target given context
            
        writer.add_scalar(f'{title}/LL', torch.Tensor(LL).mean().item(), epoch)
        writer.add_scalar(f'{title}/Recon_LL', torch.Tensor(Recon_LL).mean().item(), epoch)
        writer.add_scalar(f'{title}/Pred_LL', torch.Tensor(Pred_LL).mean().item(), epoch)
        writer.add_scalar(f'{title}/LL(STD)', torch.Tensor(LL).std().item(), epoch)
        writer.add_scalar(f'{title}/Recon_LL(STD)', torch.Tensor(Recon_LL).std().item(), epoch)
        writer.add_scalar(f'{title}/Pred_LL(STD)', torch.Tensor(Pred_LL).std().item(), epoch)        

        test_LL.append(torch.Tensor(LL).mean().item())
        test_Recon_LL.append(torch.Tensor(Recon_LL).mean().item())
        test_Pred_LL.append(torch.Tensor(Pred_LL).mean().item())
        test_g_iter.append(epoch)
        
        if fixed_img == True:
            data_test_fix = torch.load('./data_test.pt')['data_test'] #TODO
            (x_c, y_c), x_t = data_test_fix.query
            y_t = data_test_fix.target_y
            x_c = x_c.cuda()
            y_c = y_c.cuda()
            x_t = x_t.cuda()
            y_t = y_t.cuda()
            x_t = torch.cat([x_t, x_c], dim=1)
            y_t = torch.cat([y_t, y_c], dim=1)

            if args.model in {'NVDP', 'NVDP+det'}:
                y_pred = model(x_c, y_c, x_t, y_t)

            elif args.model in {'NP', 'NP+det', 'CNP'}:
                y_pred, _, _ = model(x_c, y_c, x_t, y_t)

            num_image = x_t.size(0)
        else:
            num_image = 1 

        for j in range(num_image):
            # Lets order the result, for plotting
            target_inds = torch.argsort(x_t[j, :, 0])
            x_t_sorted = x_t[j, target_inds, :].unsqueeze(0)
            y_t_sorted = y_t[j, target_inds, :].unsqueeze(0)

            context_inds = torch.argsort(x_c[j, :, 0])
            y_c_sorted = y_c[j, context_inds, :].unsqueeze(0)
            x_c_sorted = x_c[j, context_inds, :].unsqueeze(0)

            m_y_sorted = y_pred[0][j, target_inds, :].unsqueeze(0)
            s_y_sorted = y_pred[1][j, target_inds, :].unsqueeze(0)

            fig = plt.figure()
            plt.title(f"{title}. Graphic {j}")
            plot_functions(x_t_sorted.detach().cpu().numpy(),
                           y_t_sorted.detach().cpu().numpy(),
                           x_c_sorted.detach().cpu().numpy(),
                           y_c_sorted.detach().cpu().numpy(),
                           m_y_sorted.detach().cpu().numpy(),
                           s_y_sorted.detach().cpu().numpy())
            writer.add_figure(f"{title}_{str(j)}", fig, global_step=epoch, close=True)


######################################## main ########################################
print("Training Start:" + args.exp)
for epoch in range(1, args.epochs +1): 
    train(epoch)

    if epoch in args.test_interval:
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
        test(epoch, n_test=30)
    writer.flush()

####################################### Final Validation ###############################
print("Validing Start:" + args.exp)
test(epoch, 50000, title='val', fixed_img=True) # Validating with 50000 tasks

print("End Experiment" + args.exp)
writer.flush()
writer.close()
args.total_time = (time.time() - start_time)/60
print(f"--- {args.total_time: 4f} minutes ---")
