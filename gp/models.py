
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal 
from torch.distributions.bernoulli import Bernoulli
from torch.distributions import kl_divergence

# NP Model
class LatentModel(nn.Module):
    def __init__(self, args):
        super(LatentModel, self).__init__()

        self.x_dim = args.x_dim
        self.y_dim = args.y_dim
        self.r_dim = args.r_dim
        self.z_dim = args.z_dim
        self.h_dim = args.h_dim
        self.rev = args.rev

        ###################### Latent Encoder ###################
        self.h = nn.Sequential(
                nn.Linear(self.x_dim + self.y_dim, self.h_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU(inplace=True),      
                nn.Linear(self.h_dim, self.r_dim))

        self.z_encoder = nn.Sequential(
                nn.Linear(self.r_dim, self.h_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.h_dim, self.z_dim*2))

        #################### Deterministic Encoder #############
        self.d_encoder = nn.Sequential(
            nn.Linear(self.x_dim + self.y_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.r_dim))

        ################## Decoder ############################
        self.latent_path = args.latent_path
        self.det_path = args.det_path

        self.input_dim = self.x_dim
        if self.latent_path:
            self.input_dim += self.z_dim
        if self.det_path: 
            self.input_dim += self.r_dim

        self.output_dim = self.y_dim*2

        self.decode = nn.Sequential(
            nn.Linear(self.input_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.output_dim))

        self.apply(init_weights)

    def reparameterise(self, z):
        mu, std = z
        eps = torch.randn_like(std)
        z_sample = eps.mul(std).add_(mu)
        return z_sample
        
    def latent_encoder(self, x, y):
        x_y = torch.cat([x, y], dim=-1)
        r_i = self.h(x_y)
        r = torch.mean(r_i, dim=1)

        out = self.z_encoder(r)
        mu, logstd = torch.split(out, self.z_dim, dim=-1)
        std = 0.1 + 0.9 * torch.sigmoid(logstd)
        return mu, std

    def det_encoder(self, x_c, y_c, x_t):
        x_y = torch.cat([x_c, y_c], dim=2)
        r_i = self.d_encoder(x_y)
        r = torch.mean(r_i, dim=1, keepdim=True)
        return r

    def decoder(self, z_sample, x_target, r_context):
        concat = x_target
        if self.latent_path:
            concat = torch.cat([concat, z_sample], dim=-1)
        if self.det_path:
            concat = torch.cat([concat, r_context.repeat(1,x_target.size(1),1)], dim=-1)

        out = self.decode(concat)
        mu, logstd = torch.split(out, self.y_dim, dim=-1)
        std = 0.1 + 0.9 * F.softplus(logstd)
        return mu, std

    def forward(self, x_c, y_c, x_t=None, y_t=None):
        r  = None
        if self.det_path:
            r = self.det_encoder(x_c, y_c, x_t)

        z_c = self.latent_encoder(x_c, y_c)  # (mu, std)
        if self.training: 
            z_t = self.latent_encoder(x_t, y_t)
        else:  
            z_t = z_c

        if self.rev == False:
            z = self.reparameterise(z_t)
        else: 
            z = self.reparameterise(z_c)

        z = z.unsqueeze(1).expand(-1, x_t.size(1), -1)

        y_pred = self.decoder(z, x_t, r) # mu and sigma

        return y_pred, z_t, z_c


class ModelNVDP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.x_dim = args.x_dim
        self.y_dim = args.y_dim
        self.r_dim = args.r_dim
        self.h_dim = args.h_dim
        self.alpha = args.alpha

        self.d_encoder = nn.Sequential(
            nn.Linear(self.x_dim + self.y_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.r_dim))

        self.d_encoder2 = nn.Sequential(
            nn.Linear(self.x_dim + self.y_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.r_dim))


        ################## Decoder ############################
        self.det_path = args.det_path

        self.input_dim = self.x_dim
        if self.det_path: 
            self.input_dim += self.r_dim

        self.output_dim = self.y_dim*2

        self.g_1 = VariationalDropoutNVDPbias(self.input_dim, self.h_dim, r_dim = self.r_dim, alpha=self.alpha)
        self.g_2 = VariationalDropoutNVDPbias(self.h_dim, self.h_dim, r_dim = self.r_dim, alpha=self.alpha)
        self.g_3 = VariationalDropoutNVDPbias(self.h_dim, self.h_dim, r_dim = self.r_dim, alpha=self.alpha)
        self.g_4 = VariationalDropoutNVDPbias(self.h_dim, self.h_dim, r_dim =self.r_dim, alpha=self.alpha)
        self.g_5 = VariationalDropoutNVDPbias(self.h_dim, self.output_dim, r_dim =self.r_dim, alpha=self.alpha)

        self.apply(init_weights)

    def det_encoder(self, x_c, y_c, x_t):
        x_y = torch.cat([x_c, y_c], dim=2)
        r_i = self.d_encoder(x_y)
        r = torch.mean(r_i, dim=1, keepdim=True)
        return r

    def det_encoder2(self, x_c, y_c, x_t):
        x_y = torch.cat([x_c, y_c], dim=2)
        r_i = self.d_encoder2(x_y)
        r = torch.mean(r_i, dim=1, keepdim=True)
        return r

    def decoder(self, concat, r, sampling=False):

        out1 = self.g_1(concat, r, sampling)
        out2 = self.g_2(F.relu(out1), r, sampling)
        out3 = self.g_3(F.relu(out2), r, sampling)
        out4 = self.g_4(F.relu(out3), r, sampling)
        out5 = self.g_5(F.relu(out4), r, sampling)

        mu, logstd = torch.split(out5, self.y_dim, dim=-1)
        std = 0.1 + 0.9 * F.softplus(logstd)

        return mu, std

    def forward(self, x_c, y_c, x_t=None, y_t=None, sampling=False):

        concat = x_t

        if self.det_path:
            r_d = self.det_encoder(x_c, y_c, x_t)
            concat = torch.cat([concat, r_d.repeat(1, x_t.size(1), 1)], dim=-1)

        r_c = self.det_encoder2(x_c, y_c, x_t)
        y_pred = self.decoder(concat, r_c, sampling) # mu and sigma

        return y_pred


class VariationalDropoutNVDPbias(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, r_dim=32, tau=1.0, gumbel=False, alpha=-1):
        super().__init__(in_features, out_features, bias)

        self.eps = 1e-10
        self.r_dim = r_dim
        self.in_features = in_features
        self.out_features = out_features
        self.gumbel = gumbel
        self.alpha = alpha
        self.tau = torch.nn.Parameter(torch.ones(1)*1.5, requires_grad=True)
                
        self.log_alpha_net = nn.Sequential(
                nn.Linear(self.r_dim, self.r_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(self.r_dim, self.r_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(self.r_dim, self.r_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(self.r_dim, self.r_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(self.r_dim, in_features+out_features+1+out_features),
        )
        
        self.apply(init_weights)
        
    def kl2(self, r_t):
        p1 = torch.cat([self.p.view(-1), self.p_bias.view(-1)],-1)
        log_alpha = self.log_alpha_net(r_t).clamp(min=-4, max=4)/self.tau.clamp(0.5, 5)
        log_alpha1, log_alpha2, log_alpha3, log_alpha_bias = torch.split(log_alpha, [self.in_features, self.out_features, 1, self.out_features], -1)
        p = torch.sigmoid(log_alpha1) * torch.sigmoid(log_alpha2.permute(0,2,1)) * torch.sigmoid(log_alpha3)    #poe
        p_bias = torch.sigmoid(log_alpha_bias)
        p2 = torch.cat([p.view(-1), p_bias.view(-1)], -1)

        kld = 0.5 * ((p1*(1-p1)+(p2-p1)**2) / (p2*(1-p2)+self.eps) + (torch.log(p2*(1-p2)+self.eps) - torch.log(p1*(1-p1)+self.eps)) - 1)

        return torch.mean(kld)

    def forward(self, input, r, sampling=False):
        log_alpha = self.log_alpha_net(r).clamp(min=self.alpha, max=2.5)/self.tau.clamp(0.5, 5)

        log_alpha1, log_alpha2, log_alpha3, log_alpha_bias = torch.split(log_alpha, [self.in_features, self.out_features, 1, self.out_features], -1)
        self.p = torch.sigmoid(log_alpha1) * torch.sigmoid(log_alpha2).permute(0,2,1) * torch.sigmoid(log_alpha3)
        self.p_bias = torch.sigmoid(log_alpha_bias)
    
        if self.training: # Local repraramterization applied

            w_mean = (1.0-self.p) * self.weight.unsqueeze(0) 
            b_mean = (1.0-self.p_bias) * self.bias
             
            wb_mean = (input).bmm(w_mean.permute(0,2,1)) + b_mean

            w_var = self.p*(1.0 - self.p) * self.weight.unsqueeze(0)**2
            b_var = self.p_bias*(1.0 - self.p_bias) * self.bias**2

            wb_std = torch.sqrt( (input**2).bmm(w_var.permute(0,2,1)) + b_var + self.eps) 
            eps2 = torch.randn(*wb_mean.size()).to(input.device)

            return wb_mean + wb_std * eps2

        # elif sampling: # Sampling Weight
        #     aweight1 = (1.0-self.p) * self.weight.unsqueeze(0) 
        #     aweight2 = self.p*(1.0 -self.p) * self.weight.unsqueeze(0)**2
        #     aweight = aweight1 + torch.sqrt(aweight2 + self.eps) * torch.randn(*self.weight.size()).to(input.device) #* 1.3
        #     b_sample = (input).bmm(aweight.permute(0,2,1)) + self.bias 
        #     return b_sample

        else: # Output Expectation in test
            w_mean = (1.0-self.p) * self.weight.unsqueeze(0)
            b_mean = (1.0-self.p_bias) * self.bias
            wb_mean = (input).bmm(w_mean.permute(0,2,1)) + b_mean
            return wb_mean

###################### Loss  ##############################
# def kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p):
#     kl_div = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2)/torch.exp(logvar_p) - 1.0  + logvar_p - logvar_q
#     kl_div = 0.5 * kl_div.sum()
#     return kl_div

# def correct_kl(mu1, logvar1, mu2, logvar2):
#     kl_div = 0.5 * ((torch.exp(logvar1) + (mu1 - mu2)**2)/torch.exp(logvar2) - 1.0 + logvar2 - logvar1)
#     return kl_div.sum(-1).mean()

# def kl_div_gaussians2(mu_q, logvar_q, mu_p, logvar_p):
#     q_target_dist = Normal(mu_q, torch.exp(0.5*logvar_q))
#     p_context_dist = Normal(mu_p, torch.exp(0.5*logvar_p))
#     kl_div = kl_divergence(q_target_dist, p_context_dist).sum(-1).mean()
#     return kl_div

def kl_div_gaussians3(mu_q, std_q, mu_p, std_p):
    q_target_dist = Normal(mu_q, std_q)
    p_context_dist = Normal(mu_p, std_p)
    kl_div = kl_divergence(q_target_dist, p_context_dist).sum(-1).mean()
    return kl_div

def bce_loss2(y_hat, y_t):
    bce = F.binary_cross_entropy(y_hat, y_t) ## / y_t.numel()
    return bce

def bce_loss(y_hat, y_t):
    y_dist = Bernoulli(probs = y_hat)
    bce = -y_dist.log_prob(y_t)
    return bce.mean()

def log_prob(value, loc, log_scale): # Gaussian Logprobability with logvar
    var = torch.exp(log_scale * 2)
    return ( -((value - loc) ** 2)/(2*var) -log_scale -torch.log(torch.sqrt(2*torch.pi)) )

@torch.no_grad()
def init_weights(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None: 
            #m.bias.fill_(0.01)
            nn.init.normal_(m.bias, mean=0, std=1e-3)

@torch.no_grad()
def renorm_weights(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'): 
            norms = torch.sqrt(m.weight**2)
            m.weight.div(norms/8 + 1e-12)
        if hasattr(m, 'bias'): 
            norms = torch.sqrt(m.bias**2)
            m.bias.div(norms/8 + 1e-12)

def kl_div2(model, r_t):
    kl = 0
    numl = 0
    for module in model.children():
        if hasattr(module, 'kl2'): 
            kl += module.kl2(r_t)
            numl += 1.0
    return kl / numl
