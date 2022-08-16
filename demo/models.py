import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelNVDP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.x_dim = args.x_dim
        self.y_dim = args.y_dim
        self.r_dim = args.r_dim
        self.h_dim = args.h_dim
        self.alpha = args.alpha

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
        self.input_dim = self.x_dim + self.r_dim
        self.output_dim = self.y_dim*2

        self.g_1 = VariationalDropoutNVDPbias(self.input_dim, self.h_dim, r_dim = self.r_dim, alpha=self.alpha)
        self.g_2 = VariationalDropoutNVDPbias(self.h_dim, self.h_dim, r_dim = self.r_dim, alpha=self.alpha)
        self.g_3 = VariationalDropoutNVDPbias(self.h_dim, self.h_dim, r_dim = self.r_dim, alpha=self.alpha)
        self.g_4 = VariationalDropoutNVDPbias(self.h_dim, self.output_dim, r_dim =self.r_dim, alpha=self.alpha)

        self.apply(init_weights)

    def det_encoder(self, x_c, y_c, x_t):
        x_y = torch.cat([x_c, y_c], dim=2)
        r_i = self.d_encoder(x_y)
        r = r_i.mean(1, keepdim=True)
        #r_i = r_i.unsqueeze(1).expand(-1, x_t.size(1), -1)
        return r
    
    def decoder(self, x_t, r_c, sampling=False):
        concat = torch.cat([x_t, r_c], dim=-1)
        
        r = r_c.mean(1, keepdim=True)
        out = F.relu(self.g_1(concat, r, sampling))
        out = F.relu(self.g_2(out, r, sampling))
        out = F.relu(self.g_3(out, r, sampling))
        out = self.g_4(out, r, sampling)

        mu, logstd = torch.split(out, self.y_dim, dim=-1)
        std = F.softplus(logstd)
        std = 0.1 + 0.9 * F.softplus(logstd)
        return mu, std

    def forward(self, x_c, y_c, x_t=None, y_t=None):
        r = self.det_encoder(x_c, y_c, x_t)
        r = r.expand(-1, x_t.size(1), -1)

        y_pred = self.decoder(x_t, r) # mu and sigma
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

        else: # Output Expectation in test
            w_mean = (1.0-self.p) * self.weight.unsqueeze(0)
            b_mean = (1.0-self.p_bias) * self.bias
            wb_mean = (input).bmm(w_mean.permute(0,2,1)) + b_mean
            return wb_mean


        
###################### Utilities ##############################

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



def pmap(model, r=None):
    pmap = []
    wmap = []
    for module in model.children():
        if hasattr(module, 'kl2'):

            log_alpha = module.log_alpha_net(r)
            log_alpha1, log_alpha2, log_alpha3, log_alpha_bias = torch.split(log_alpha, [module.in_features, module.out_features, 1, module.out_features], -1)
            p = torch.sigmoid(log_alpha1) * torch.sigmoid(log_alpha2.permute(0,2,1)) * torch.sigmoid(log_alpha3)    #poe
            pmap.append((1-p).detach().cpu())
            wmap.append(module.weight.detach().cpu())

    return pmap, wmap

