'''
This is a minimal adaptation of LeWorldModel to online setting in the Atari environment. 
The dynamics model is learnt in spirit with LeWorldModel: 
a CNN encoder encodes observations into embeddings and a gru based predictor predicts the embedding of the current observation conditioned on previous observations and actions. 
The encoder and predictor are trained using prediction and sigreg losses.
The behaviour model is learnt using the Dreamer-v3 famerwork: 
a reward model and a discount model are learnt as probes over the learnt dynamics model in a detached state. 
Actor and Critic models are trained using imagined rollouts over the learnt dynamics model.

'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import matplotlib.pyplot as plt
from copy import deepcopy
import gymnasium as gym
from tqdm import tqdm
import imageio
import cv2
import ale_py


# --------------------- distribution utils: oneHot and twoHot ------------------ #

def to_f32(x):
    return x.to(dtype=torch.float32)


def to_i32(x):
    return x.to(dtype=torch.int32)


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


# overwrite oneHotDist allowing for unimix
class OneHotDist(tdist.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits, unimix_ratio=0.0):
        # (..., K)
        probs = F.softmax(to_f32(logits), dim=-1)
        uniform = unimix_ratio / probs.shape[-1]
        probs = probs * (1.0 - unimix_ratio) + torch.ones_like(probs, dtype=torch.float32) * uniform
        logits = torch.log(probs)
        super().__init__(logits=logits)

    @property
    def mode(self):
        # (..., K)
        _mode = F.one_hot(torch.argmax(self.logits, axis=-1), self.logits.shape[-1])
        return _mode.detach() + self.logits - self.logits.detach()

    def rsample(self, sample_shape=(), temperature=1.0):
        # (..., K)
        return F.gumbel_softmax(self.logits, tau=temperature, hard=True, dim=-1)
        # return self.sample() + self.probs - self.probs.detach()

    def sample(self, **kwargs):
        raise NotImplementedError


class TwoHot:
    def __init__(self, logits, bins, squash=None, unsquash=None):
        # (..., N_bins), (N_bins,)
        self.logits = to_f32(logits)
        assert self.logits.shape[-1] == len(bins), (self.logits.shape, len(bins))

        self.bins = bins
        self.probs = F.softmax(self.logits, dim=-1)  # (..., N_bins)
        self.squash = squash if squash is not None else (lambda x: x)
        self.unsquash = unsquash if unsquash is not None else (lambda x: x)

    def mode(self):
        # (..., N_bins), (N_bins,) -> (..., 1)
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m : m + 1]
            p3 = self.probs[..., m + 1 :]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m : m + 1]
            b3 = self.bins[..., m + 1 :]
            wavg = (p2 * b2).sum(dim=-1, keepdim=True) + ((p1 * b1).flip(dims=(-1,)) + (p3 * b3)).sum(
                dim=-1, keepdim=True
            )
            return self.unsquash(wavg)
        p1 = self.probs[..., : n // 2]
        p2 = self.probs[..., n // 2 :]
        b1 = self.bins[..., : n // 2]
        b2 = self.bins[..., n // 2 :]
        wavg = ((p1 * b1).flip(dims=(-1,)) + (p2 * b2)).sum(dim=-1, keepdim=True)
        return self.unsquash(wavg)

    def log_prob(self, target):
        # (..., 1)
        assert target.dtype == self.probs.dtype
        target = target.squeeze(-1)  # (...,)
        target_squashed = self.squash(target).detach()  # (...,)
        # below/above: (...,)
        below = to_i32(self.bins <= target_squashed.unsqueeze(-1)).sum(dim=-1) - 1
        above = len(self.bins) - to_i32(self.bins > target_squashed.unsqueeze(-1)).sum(dim=-1)
        below = torch.clamp(below, 0, len(self.bins) - 1)
        above = torch.clamp(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[below] - target_squashed).abs(),
        )
        dist_to_above = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[above] - target_squashed).abs(),
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        oh_below = to_f32(F.one_hot(below, num_classes=len(self.bins)))
        oh_above = to_f32(F.one_hot(above, num_classes=len(self.bins)))
        # (..., N_bins)
        mixed_target = oh_below * weight_below.unsqueeze(-1) + oh_above * weight_above.unsqueeze(-1)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)  # (..., N_bins)
        return (mixed_target * log_pred).sum(dim=-1)  # (...)


def onehot(mean, unimix_ratio, **kwargs):
    return OneHotDist(to_f32(mean), unimix_ratio=unimix_ratio)


def symexp_twohot(logits, bin_num, **kwargs):
    if bin_num % 2 == 1:
        half = torch.linspace(-20, 0, (bin_num - 1) // 2 + 1, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half[:-1].flip(dims=(0,))], 0)
    else:
        half = torch.linspace(-20, 0, bin_num // 2, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half.flip(dims=(0,))], 0)
    return TwoHot(to_f32(logits), bins)


# class implementing sigreg
class SIGReg(nn.Module):
    def __init__(self, device, n_proj, knots=17, max_t=3):
        super().__init__()
        self.n_proj = n_proj 
        self.t = torch.linspace(0, max_t, knots).to(device)
        self.dt = max_t / (knots - 1)
        self.weights = torch.full((knots,), 2 * self.dt).to(device) # [t]
        self.weights[[0, -1]] = self.dt # endpoints should get half weighting in accordance with torch.trapz  
        self.target_cf = torch.exp(-self.t.square() / 2.0).to(device) # [t]
        self.weights = self.target_cf * self.weights # [t]

    def forward(self, emb): # emb.shape: [T, N, d]
        assert len(emb.shape) == 3, f'{len(emb.shape)} != 3 for sigreg'
        T, N, d = emb.shape
        A = torch.randn(d, self.n_proj).to(emb.device) # [d, p]
        A = A / A.norm(p=2, dim=0, keepdim=True)
        x = torch.matmul(emb, A) # [T, N, p]
        x = x.unsqueeze(-1) * self.t # [T, N, p, t]
        empirical_cf_real, empirical_cf_imaginary = x.cos().mean(dim=-3), x.sin().mean(dim=-3) # [T, p, t]
        error = (empirical_cf_real - self.target_cf).square() + empirical_cf_imaginary.square() # [T, p, t]
        integrated = torch.matmul(error, self.weights.unsqueeze(-1)) # [T, p, 1]
        statistic = integrated * N 
        return statistic.sum(dim=0).mean()
    
# --------------------- define networks ------------------ #

# batchnorm projection head 
class Batchnorm_ProjHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        h_dim = out_dim * 2
        self.proj_head = nn.Sequential(
                        nn.Linear(in_dim, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.GELU(),

                        nn.Linear(h_dim, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.GELU(),

                        nn.Linear(h_dim, out_dim)
        )

    def forward(self, x):
        return self.proj_head(x)


# residual conv2d block (no downsampling)
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, channels//8), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, channels//8), channels),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))  # residual connection


# observation encoder
class ObsEncoder(nn.Module):
    def __init__(self, input_channels, h_dim, out_dim, dropout):
        super().__init__()

        def make_stage(in_ch, out_ch):
            return nn.Sequential(
                # Separate feature learning from downsampling
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(max(1, out_ch // 8), out_ch),
                nn.SiLU(),
                ResBlock(out_ch),          # residual refinement at this scale
                nn.AvgPool2d(2),           # clean spatial downsampling
            )

        self.net = nn.Sequential(
            make_stage(input_channels,  8),
            make_stage(8,              16),
            make_stage(16,             32),
            make_stage(32,             64),
        )

        latent_size = img_size // 16
        self.fc = nn.Sequential(
            nn.Linear(64 * latent_size * latent_size, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.proj_head = Batchnorm_ProjHead(h_dim, out_dim)

    def forward(self, x):
        batch_shape = x.shape[:-3]
        x = x.flatten(start_dim=0, end_dim=-4)
        x = self.net(x)
        x = x.reshape(*batch_shape, -1)
        x = self.fc(x)
        emb = self.proj_head(x)
        return emb


# RSSM used for both encoder s_t = f(o_t) and the predictor s_t = g(s_t-1, a_t-1)
# for now, f is just a CNN and g is a gru
class RSSM(nn.Module):
    def __init__(self, n_channels, a_dim, s_dim, belief_dim, h_dim, batch_size, device):
        super().__init__()

        # observation encoder
        self.encoder = ObsEncoder(n_channels, h_dim, s_dim, dropout=0.1)

        # action encoder 
        self.a_encoder = nn.Sequential(
                            nn.Linear(a_dim, h_dim),
                            nn.SiLU(),
                            nn.Linear(h_dim, s_dim),
                            nn.LayerNorm(s_dim),
        )

        # predictor
        self.predictor_gru = nn.GRUCell(s_dim * 2, belief_dim)
        self.predictor_net = nn.Sequential(
                                nn.Linear(belief_dim, h_dim),
                                nn.SiLU(),
                                nn.Linear(h_dim, h_dim),
                                nn.LayerNorm(h_dim),
        )
        self.predictor_proj_head = Batchnorm_ProjHead(h_dim, s_dim)

        self.device = device

    def get_encoded_state(self, obs):
        return self.encoder(obs)
    
    def get_predicted_state(self, prev_belief, prev_state, prev_action):
        prev_action_encoded = self.a_encoder(prev_action)
        gru_input = torch.cat([prev_state, prev_action_encoded], dim=-1)
        belief = self.predictor_gru(gru_input, prev_belief)
        x = self.predictor_net(belief)
        state = self.predictor_proj_head(x)
        return belief, state 

    # forward pass through RSSM
    def forward(self, prev_belief, prev_state, prev_action, obs):
        enc_state = self.get_encoded_state(obs)
        belief, pred_state = self.get_predicted_state(prev_belief, prev_state, prev_action)
        return belief, pred_state, enc_state  



# Observation Decoder (Gaussian with deconv layers)
class Observation_Model_Deconv_Gaussian(nn.Module):
    def __init__(self, in_dim, h_dim, n_classes):
        super().__init__()

        self.n_classes = n_classes
        self.latent_size = img_size // 16

        self.fc = nn.Linear(in_dim, 64 * self.latent_size * self.latent_size)
        self.silu = nn.SiLU()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  
            nn.SiLU(),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  
            nn.SiLU(),

            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),   
            nn.SiLU(),

            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),    
            nn.SiLU(),

            # final conv (no stride change)
            nn.Conv2d(4, n_classes * 2, kernel_size=3, padding=1)
        )

    def forward(self, state, belief):
        x = torch.cat([state, belief], dim=-1)
        h = self.fc(x)

        batch_shape = h.shape[:-1]
        h = h.flatten(start_dim=0, end_dim=-2)
        h = h.view(h.shape[0], 64, self.latent_size, self.latent_size) # [B, 64, 4, 4]

        out = self.net(h)  # [B, c*2, h, w]

        mean, logstd = out[:, :self.n_classes], out[:, self.n_classes:] # [B, c, h, w]
        logstd = logstd.clip(minClip, maxClip)
        std = torch.exp(logstd)

        return mean, std
    

    # to draw sample from the learnt probabilistic model
    def sample(self, state, belief):
        batch_shape = state.shape[:-1]
        mean, std = self.forward(state, belief)
        eps = torch.randn_like(std)
        out = mean + eps * std # [B, c, h, w]
        out = out.permute(0,2,3,1) # [B, h, w, c]
        # restore batch dims 
        out = out.reshape(*batch_shape, *out.shape[-3:]) # [b_dims, h, w, c]
        return out

    # calculates log p(y|x)
    def log_prob(self, state, belief, y): # y.shape = [b_dims, c, h, w]
        batch_shape = state.shape[:-1]
        y = y.flatten(-3, -1) # [b_dims, c*h*w]

        mean, std = self.forward(state, belief) # [B, c, h, w]

        # restore batch_shape
        mean = mean.reshape(*batch_shape, *mean.shape[-3:]) # [b_dims, c, h, w]
        std = std.reshape(*batch_shape, *std.shape[-3:])

        # flatten c, h, w
        mean = mean.flatten(-3, -1) # [b_dims, c*h*w]
        std = std.flatten(-3, -1)

        dis = tdist.independent.Independent(tdist.Normal(mean, std), 1)
        lp = dis.log_prob(y) # [b_dims]
        
        return lp


# Discount Model - parameterized bernouli dist
class Discount_Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, out_dim)
        self.silu = nn.SiLU()

    # forward pass through the stochastic net
    def forward(self, state, belief):
        x = torch.cat([state, belief], dim=-1)
        h = self.silu(self.fc1(x))
        h = self.silu(self.fc2(h))
        logits = self.fc3(h)
        return logits

    # to draw sample from the learnt probabilistic model
    def sample(self, state, belief):
        logits = self.forward(state, belief)
        dis = tdist.Bernoulli(logits=logits)
        out = dis.sample()
        # for straight through gradient
        out = out + dis.probs - dis.probs.clone().detach()
        return out

    # calculates log p(y|x)
    def log_prob(self, state, belief, y):
        logits = self.forward(state, belief)
        dis = tdist.independent.Independent(tdist.Bernoulli(logits=logits), 1)
        lp = dis.log_prob(y)
        return lp
    
    # calculate mean - used during imagination rollout
    def mean(self, state, belief):
        logits = self.forward(state, belief)
        dis = tdist.independent.Independent(tdist.Bernoulli(logits=logits), 1)
        return dis.mean 
    

# Reward model - parameterized symexp_twohot
class Reward_Model(nn.Module):
    def __init__(self, in_dim, h_dim, n_bins):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, n_bins)
        self.silu = nn.SiLU()
        self.n_bins = n_bins

        # init last layer weights to zero
        with torch.no_grad():
            self.fc3.weight.zero_()
            self.fc3.bias.zero_()

    # forward pass through the stochastic net
    def forward(self, state, belief):
        x = torch.cat([state, belief], dim=-1)
        h = self.silu(self.fc1(x))
        h = self.silu(self.fc2(h))
        logits = self.fc3(h)
        return logits
    
    # prepare symexp_twohot dist
    def get_dist(self, state, belief):
        logits = self.forward(state, belief)
        dis = symexp_twohot(logits, bin_num=self.n_bins)
        return dis

    # get mode - used during imagination rollout
    def mode(self, state, belief):
        dis = self.get_dist(state, belief)
        mode = dis.mode()
        return mode

    # calculates log p(y|x)
    def log_prob(self, state, belief, y):
        dis = self.get_dist(state, belief)
        lp = dis.log_prob(y)
        return lp


# Critic model - parameterized symexp_twohot
class Critic(nn.Module):
    def __init__(self, in_dim, h_dim, n_bins):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, n_bins)
        self.silu = nn.SiLU()
        self.n_bins = n_bins

        # init last layer weights to zero
        with torch.no_grad():
            self.fc4.weight.zero_()
            self.fc4.bias.zero_()

    def forward(self, state, belief):
        x = torch.cat([state, belief], dim=-1)
        h = self.silu(self.fc1(x))
        h = self.silu(self.fc2(h))
        h = self.silu(self.fc3(h))
        val = self.fc4(h)
        return val
    
    # prepare symexp_twohot dist
    def get_dist(self, state, belief):
        logits = self.forward(state, belief)
        dis = symexp_twohot(logits, bin_num=self.n_bins)
        return dis

    # get mode - used during imagination rollout
    def mode(self, state, belief):
        dis = self.get_dist(state, belief)
        mode = dis.mode()
        return mode

    # calculates log p(y|x)
    def log_prob(self, state, belief, y):
        dis = self.get_dist(state, belief)
        lp = dis.log_prob(y)
        return lp
    


# actor network - parameterizing one-hot categorical
class Actor(nn.Module):
    def __init__(self, in_dim, a_dim, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, h_dim)
        self.fc5_logits = nn.Linear(h_dim, a_dim)
        self.silu = nn.SiLU()

    # returns the logits of one_hot_categorical distribution representing the policy
    def forward(self, state, belief):
        x = torch.cat([state, belief], dim=-1)
        h = self.silu(self.fc1(x))
        h = self.silu(self.fc2(h))
        h = self.silu(self.fc3(h))
        h = self.silu(self.fc4(h))
        logits = self.fc5_logits(h)
        return logits

    # returns the policy as one_hot_categorical distribution
    def policy_dist(self, state, belief):
        logits = self.forward(state, belief)
        dis = OneHotDist(logits=logits, unimix_ratio=0.01) 
        return dis

    # returns policy log_prob
    def policy_logprob(self, state, belief, y):
        policy_dis = self.policy_dist(state, belief)
        lp = policy_dis.log_prob(y)
        return lp


# used to calculate scale for actor loss
class ReturnEMA(nn.Module):

    def __init__(self, device, alpha=1e-2):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)
        self.register_buffer("ema_vals", torch.zeros(2, dtype=torch.float32, device=self.device))

    def __call__(self, x):
        x_quantile = torch.quantile(torch.flatten(x.detach()), self.range)
        # Using out-of-place update for torch.compile compatibility
        self.ema_vals.copy_(self.alpha * x_quantile.detach() + (1 - self.alpha) * self.ema_vals)
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        offset = self.ema_vals[0]
        return offset.detach(), scale.detach()



# replay buffer
class ReplayBuffer:
    def __init__(self, buf_size, seq_len, batch_size, o_dim, a_dim, device):
        self.buf_size = buf_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.buf_observation = np.zeros((buf_size, *o_dim))
        self.buf_action = np.zeros((buf_size, a_dim))
        self.buf_reward = np.zeros((buf_size, 1))
        self.buf_done = np.zeros((buf_size, 1))
        self.n_items = 0
        self.device = device

    def add(self, oar_tuple):
        observation, action, reward, done = oar_tuple
        index = self.n_items % self.buf_size
        self.buf_observation[index] = observation
        self.buf_action[index] = action
        self.buf_reward[index] = reward
        self.buf_done[index] = done
        self.n_items += 1

    def sample(self):

        idx_list = []
        curr_idx = self.n_items % self.buf_size

        high = self.n_items - self.seq_len
        low = 0
        while (len(idx_list) < self.batch_size):
            idx_start = np.random.randint(low, high)
            idx_chunk = np.arange(idx_start, idx_start+self.seq_len) % self.buf_size
            # don't append sample chunks that are part old and part new
            if not (curr_idx in idx_chunk):
                idx_list.append(idx_chunk)

        idx = np.array(idx_list)
        idx = idx.T # first dimension should be time_step and second dimension should be batch
        observation = torch.FloatTensor(self.buf_observation[idx]).to(self.device)
        action = torch.FloatTensor(self.buf_action[idx]).to(self.device)
        reward = torch.FloatTensor(self.buf_reward[idx]).to(self.device)
        done = torch.FloatTensor(self.buf_done[idx]).to(self.device)
        return (observation, action, reward, done)



# LEWM_Dreamerv3
class LEWM_Dreamerv3(nn.Module):
    def __init__(self, o_dim, s_dim, belief_dim, a_dim, h_dim, seq_len, imagination_horizon, df, buf_size, batch_size, lr_actor, lr_critic, lr_model, _lambda, tau, rho, eta, n_bins, n_channels, sigreg, sigreg_lambda, device):
        super().__init__()
        self.actor = Actor(s_dim + belief_dim, a_dim, h_dim).to(device)
        self.critic_V = Critic(s_dim + belief_dim, h_dim, n_bins).to(device)
        self.target_critic_V = deepcopy(self.critic_V)
        self.replay_buffer = ReplayBuffer(buf_size, seq_len, batch_size, o_dim, a_dim, device)
        self.rssm = RSSM(n_channels, a_dim, s_dim, belief_dim, h_dim, batch_size, device).to(device)
        self.df_model = Discount_Model(s_dim + belief_dim, h_dim, 1).to(device)
        self.reward_model = Reward_Model(s_dim + belief_dim, h_dim, n_bins).to(device)
        self.observation_model = Observation_Model_Deconv_Gaussian(s_dim + belief_dim, h_dim, n_channels).to(device)
        self.return_ema = ReturnEMA(device).to(device)
        self.optimizer_actor = torch.optim.Adam(params=self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic_V = torch.optim.Adam(params=self.critic_V.parameters(), lr=lr_critic)
        self.optimizer_model = torch.optim.Adam(params=list(self.rssm.parameters()) + list(self.df_model.parameters()) + \
        list(self.reward_model.parameters()) + list(self.observation_model.parameters()), lr=lr_model)
        self.df = df
        self.s_dim = s_dim
        self.belief_dim = belief_dim
        self.a_dim = a_dim
        self.o_dim = o_dim
        self.device = device
        self.train_iters = 0
        self.tanh = nn.Tanh()
        self.seq_len = seq_len
        self.imagination_horizon = imagination_horizon
        self._lambda = _lambda
        self.tau = tau # used when updating target_critic_V
        self.rho = rho # used for weighing actor dynamics loss and actor reinforce loss
        self.eta = eta # used for weighing entropy regulaization in actor loss
        self.batch_size = batch_size
        self.sigreg = sigreg
        self.sigreg_lambda = sigreg_lambda

    def get_action(self, state, belief):
        policy = self.actor.policy_dist(state, belief)
        action = policy.rsample() # gumbel-softmax sample is differentiable
        # # for straight through gradient
        # action = action + policy.probs - policy.probs.clone().detach()
        return action, policy

    # epsilon greedy exploration
    def action_exploration(self, action, epsilon):
        if torch.rand(1) < epsilon:
            idx = torch.randint(low=0, high=self.a_dim, size=(1,))
            action = torch.zeros(1, self.a_dim)
            action[0][idx] = 1
        return action


    def freeze_model_params(self, model):
        for param in model.parameters():
            param.requires_grad_(False)

    def unfreeze_model_params(self, model):
        for param in model.parameters():
            param.requires_grad_(True)


    def calculate_lambda_return(self, rewards, discounts, state_values_target):
        """
        Input:
        # rewards obtained from reward model - r[t+1 : t+H+1]
        # df values obtained from df model - df[t+1 : t+H+1]
        # state values obtained from target_critic model - v[t+1 : t+H+1]

        Output:
        # V_lambda[t+1 : t+H]
        """
        lambda_returns = []

        # prepare accumulator bootstrap value 
        accumulator = state_values_target[-1] # v[t+H+1]

        # adjust tensors
        rewards = rewards[1:] # r[t+2 : t+H+1]
        discounts = discounts[1:] # df[t+2 : t+H+1]
        state_values_target = state_values_target[1:] # v[t+2 : t+H+1]

        for t in range(len(rewards)-1, -1, -1): # t is just going from last element to first element, since all arrays are of length H
            accumulator = rewards[t] + discounts[t] * ( (1 - self._lambda)*state_values_target[t] + self._lambda*accumulator )
            # V_lambda[j] = reward[j+1] + df[j+1] * ( (1-lambda) * V_t[j+1] + lambda * V_lambda[j+1] )
            lambda_returns = [accumulator] + lambda_returns
        lambda_returns = torch.stack(lambda_returns, dim=0) # V_lambda[t+1 : t+H]
        return lambda_returns


    def train(self):

        # sample from replay buffer
        observation, action, reward, done = self.replay_buffer.sample() 
        observation = observation.permute(0,1,4,2,3) # [H, b, c, h, w]

        #########################
        ## dynamics learning (using experience sampled from replay buffer)
        #########################

        # unfreeze dynamics model params
        self.unfreeze_model_params(self.rssm)
        self.unfreeze_model_params(self.df_model)
        self.unfreeze_model_params(self.reward_model)
        self.unfreeze_model_params(self.observation_model)

        # list tensors to store
        belief_list = [] 
        pred_state_list = []
        enc_state_list = []

        # fixed prev_belief, prev_state and prev_action (to intialize the first state)
        prev_belief = torch.zeros(self.batch_size, self.belief_dim).to(self.device)
        prev_state = torch.zeros(self.batch_size, self.s_dim).to(self.device)
        prev_action = torch.zeros(self.batch_size, self.a_dim).to(device)

        # rssm rollout 
        for t in range(observation.shape[0] - 1): # [t : t+H-1] 

            # reset if done
            if t > 0:
                prev_belief = prev_belief * (1. - done[t-1])
                prev_state = prev_state * (1. - done[t-1])
                prev_action = prev_action * (1. - done[t-1])

            # get curr state (encoded and predicted)
            belief, pred_state, enc_state = self.rssm(prev_belief, prev_state, prev_action, observation[t])

            belief_list.append(belief)
            pred_state_list.append(pred_state)
            enc_state_list.append(enc_state)

            # for next step (no detach)
            prev_belief = belief # memory element representing trajectory history
            prev_state = enc_state # teacher-forcing 
            prev_action = action[t+1] # since a_t in replay buffer is action taken to reach o_t


        # rollout ended - stack lists into tensors
        beliefs = torch.stack(belief_list, dim=0) # [t : t+H-1] 
        pred_states = torch.stack(pred_state_list, dim=0) # [t : t+H-1] 
        enc_states = torch.stack(enc_state_list, dim=0) # [t : t+H-1]


        ## calculate loss terms 

        # reward loss
        lp_reward = self.reward_model.log_prob(enc_states.detach().clone(), beliefs.detach().clone(), reward[:-1]) # logp( r[t:t+H-1] | s[t:t+H-1], h[t:t+H-1] )
        lp_reward = lp_reward.sum(dim=0).mean()

        # observation loss (reconstruction)
        lp_obs = self.observation_model.log_prob(enc_states.detach().clone(), beliefs.detach().clone(), observation[:-1]) # logp( o[t:t+H-1] | s[t:t+H-1], h[t:t+H-1] )
        lp_obs = lp_obs.sum(dim=0).mean() 

        # df loss
        lp_df = self.df_model.log_prob(enc_states.detach().clone(), beliefs.detach().clone(), (1. - done[:-1])) # logp( d[t:t+H-1] | s[t:t+H-1], h[t:t+H-1] )
        lp_df = lp_df.sum(dim=0).mean() 
            
        # dynamics loss 
        loss_pred = (pred_states - enc_states).square().sum(dim=0).mean()
        loss_sigreg = self.sigreg(enc_states) * self.sigreg_lambda
        loss_dynamics = loss_pred + loss_sigreg

        # probes loss
        loss_probes = -1 * (lp_reward + lp_obs + lp_df) * probe_mult

        # total loss
        loss = loss_dynamics + loss_probes

        # update dynamics model
        self.optimizer_model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.rssm.parameters()) + list(self.df_model.parameters()) + list(self.reward_model.parameters()) + \
                                list(self.observation_model.parameters()) , 100., norm_type=2)
        self.optimizer_model.step()

        # loss accumulators for book keeping and plotting
        loss_reward = -lp_reward
        loss_obs = -lp_obs
        loss_df = -lp_df

        ####################
        ## behaviour learning (using imagined rollouts over the learnt dynamics model)
        ####################

        # freeze dynamics model params
        self.freeze_model_params(self.rssm)
        self.freeze_model_params(self.df_model)
        self.freeze_model_params(self.reward_model)
        self.freeze_model_params(self.observation_model)

        # list tensors to store 
        im_state_list = []
        im_belief_list = []
        log_pi_list = []
        entropy_pi_list = []

        # flatten time and batch dimension into one - for parallel imagination rollouts
        # and init imagination states with these values
        im_curr_state = torch.flatten(enc_states, start_dim=0, end_dim=1).clone().detach()
        im_curr_belief = torch.flatten(beliefs, start_dim=0, end_dim=1).clone().detach()

        # imagination rollout 
        for tau in range(self.imagination_horizon):

            # get actor action, also get policy logprob and entropy
            im_curr_action, policy = self.get_action(im_curr_state, im_curr_belief)
            curr_log_pi = policy.log_prob(torch.round(im_curr_action.detach()))
            curr_entropy_pi = policy.entropy()

            # rssm step (prediction)
            im_next_belief, im_next_state = self.rssm.get_predicted_state(im_curr_belief, im_curr_state, im_curr_action)

            # store
            im_state_list.append(im_next_state)
            im_belief_list.append(im_next_belief)
            log_pi_list.append(curr_log_pi)
            entropy_pi_list.append(curr_entropy_pi)

            # for next step (no detach)
            im_curr_belief = im_next_belief 
            im_curr_state = im_next_state

        # rollout ended - stack lists to tensors
        im_states = torch.stack(im_state_list, dim=0) # s[t+1 : t+1+H]
        im_beliefs = torch.stack(im_belief_list, dim=0) # s[t+1 : t+1+H]
        log_pi = torch.stack(log_pi_list, dim=0) # logp[t : t+H]
        entropy_pi = torch.stack(entropy_pi_list, dim=0) # entropy[t : t+H]

        # calculate rewards, discounts, state_values_target and bootstrap_value for imagined rollout 
        rewards = self.reward_model.mode(im_states, im_beliefs) # r[t+1 : t+H+1]
        discounts = self.df * self.df_model.mean(im_states, im_beliefs) # df[t+1 : t+H+1]
        state_values = self.critic_V.mode(im_states, im_beliefs).detach()  # v[t+1 : t+H+1]

        # calculate lambda returns 
        lambda_returns = self.calculate_lambda_return(rewards, discounts, state_values) # v_lambda[t+1 : t+H]

        ## calculate actor loss

        # create discount cumprods
        discounts_cumprod = torch.cumprod(discounts[:-1], dim=0).detach() # df[t+1 : t+H]

        # loss through dynamics 
        loss_actor_dynamics = -lambda_returns * discounts_cumprod
        loss_actor_dynamics = loss_actor_dynamics.sum(dim=0).mean() # sum over horizon dim and mean over batch dim

        # reinforce loss
        ret_offset, ret_scale = self.return_ema(lambda_returns)
        advantage = (lambda_returns - state_values[:-1]) / ret_scale # advantage[t+1 : t+H]
        loss_actor_reinforce = (-log_pi[1:].unsqueeze(-1) * advantage.detach()) * discounts_cumprod 
        loss_actor_reinforce = loss_actor_reinforce.sum(dim=0).mean()

        # policy entropy for regularization (and encourage exploration)
        policy_entropy = entropy_pi[1:].unsqueeze(-1) * discounts_cumprod
        policy_entropy = policy_entropy.sum(dim=0).mean()

        # total loss for actor - weight by discount factor
        loss_actor = (1 - self.rho) * loss_actor_dynamics + self.rho * loss_actor_reinforce - self.eta * policy_entropy

        ## calculate critic loss 

        target_values = self.target_critic_V.mode( im_states[:-1], im_beliefs[:-1] ).detach()

        lp_critic = self.critic_V.log_prob( im_states[:-1].detach(), im_beliefs[:-1].detach(), lambda_returns ) + \
                      self.critic_V.log_prob( im_states[:-1].detach(), im_beliefs[:-1].detach(), target_values ) 
        
        lp_critic = lp_critic.unsqueeze(-1) * discounts_cumprod

        loss_critic = -lp_critic.sum(dim=0).mean() # sum over horizon dim and mean over batch dim

        ## update actor and critic

        self.optimizer_actor.zero_grad()
        self.optimizer_critic_V.zero_grad()

        loss_actor.backward()
        loss_critic.backward()

        nn.utils.clip_grad_norm_(self.actor.parameters() , 100., norm_type=2)
        nn.utils.clip_grad_norm_(self.critic_V.parameters() , 100., norm_type=2)

        self.optimizer_actor.step()
        self.optimizer_critic_V.step()        

        return loss_dynamics, loss_pred, loss_sigreg, loss_reward, loss_obs, loss_df, loss_actor, loss_critic, policy_entropy, ret_scale


# utility function to preprocess frame 
def preprocess_frame(obs, resize, channels):
    # Grayscale
    if channels == 1:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # [210, 160]
    else:
        gray = obs # [210, 160, 3]

    # Crop score area and resize
    gray = cv2.resize(gray, (resize, resize), 
                      interpolation=cv2.INTER_AREA)  
    gray = gray.astype(np.float32) / 255.0
    gray = gray * 2 - 1                             # normalize to [-1, 1]

    gray = torch.from_numpy(gray).float().clip(-1,1)
    if channels == 1:
        gray = gray.unsqueeze(-1)
    return gray 



# main
if __name__ == '__main__':

    # hyperparams
    sigreg_proj = 16 # number of sigreg projections per batch
    sigreg_lambda = 0.1
    n_channels = 3 
    img_size = 32
    n_bins = 255
    s_dim = 256 # this is also the jepa embedding dim
    belief_dim = s_dim * 2 # belief state is just used as memory - representing the trajectory history
    h_dim = s_dim * 2
    lr_model = 3e-4
    probe_mult = 0.8
    lr_critic = lr_model * probe_mult * 0.8
    lr_actor = lr_critic * 0.5
    sample_seq_len = 32 # length of contiguous sequence sampled from replay buffer (when training)
    imagination_horizon = 16 # length of imagined rollouts using the learnt dynamics model (when behaviour learning)
    _lambda = .95 # lambda - used to calculate lambda return
    tau = 0.05 # used when updating target_critic_V
    target_critic_update_step = 1
    rho = 1 # used for weighing actor dynamics loss and actor reinforce loss
    eta = 3e-3 # used for weighing entropy regulaization in actor loss
    df = 0.997
    minClip, maxClip = -2, 2
    random_seed = 1010
    batch_size = 128
    replay_buffer_size = 10**5
    num_episodes = 300
    num_train_calls = 50
    train_episode = 1
    init_random_episodes = 5
    record_episode = num_episodes // 10
    action_repeat = 2
    explore_minLimit, explore_maxLimit = 0.05, 0.4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load environment
    game = 'Boxing-v5' # replace with any atari game listed at: https://ale.farama.org/environments/complete_list/
    env = gym.make('ALE/' + game, render_mode="rgb_array")
    a_dim = env.action_space.n
    o_dim = env.observation_space.shape
    o_dim = (img_size, img_size, n_channels)
    print('a_dim: ', a_dim)
    print('o_dim: ', o_dim)

    # hyperparam dict
    hyperparam_dict = {}
    hyperparam_dict['env'] = game
    hyperparam_dict['algo'] = 'lewm_dreamerv3_beliefMP_obsEncCNNResidual'
    hyperparam_dict['imgSz'] = str(img_size)
    hyperparam_dict['Sdim'] = str(s_dim)
    hyperparam_dict['lrModel'] = f'{lr_model:.6f}'
    hyperparam_dict['L'] = str(sample_seq_len)
    hyperparam_dict['H'] = str(imagination_horizon)
    hyperparam_dict['eta'] = str(eta)
    hyperparam_dict['tau'] = str(tau)
    hyperparam_dict['sigProj'] = str(sigreg_proj)
    hyperparam_dict['sigLambda'] = str(sigreg_lambda)
    hyperparam_dict['B'] = str(batch_size)
    hyperparam_dict['EP'] = str(num_episodes)
    hyperparam_dict['trCalls'] = str(num_train_calls)
    hyperparam_dict['trEP'] = str(train_episode)
    hyperparam_dict['probeMult'] = str(probe_mult)

    # hyperparam string
    hyperstr = ""
    for k,v in hyperparam_dict.items():
        hyperstr += k + ':' + v + "_"

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    obs, info = env.reset(seed=random_seed)
    env.action_space.seed(random_seed)

    # init sigreg 
    sigreg = SIGReg(device, sigreg_proj)

    # init agent
    agent = LEWM_Dreamerv3(o_dim, s_dim, belief_dim, a_dim, h_dim, sample_seq_len, imagination_horizon, df, replay_buffer_size, batch_size, lr_actor, lr_critic, lr_model, _lambda, tau, rho, eta, n_bins, n_channels, sigreg, sigreg_lambda, device)

    # results and stats containers
    ep_return_list = []
    loss_dynamics_list, loss_pred_list, loss_sigreg_list = [], [], []
    loss_reward_list = []
    loss_obs_list = []
    loss_df_list = []
    loss_actor_list = []
    loss_critic_list = []
    policy_entropy_list = []
    ret_scale_list = []

    # seed episodes
    for ep in range(init_random_episodes):
        obs, info = env.reset()
        obs_frame = env.render()
        done = False
        ep_steps = 0

        # first step 
        action = np.zeros(a_dim)
        reward = 0
        obs = preprocess_frame(obs_frame, img_size, n_channels)
        oar_tuple = [obs, action, reward, done]
        agent.replay_buffer.add(oar_tuple)

        while not done:
            action_scalar = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action_scalar)
            next_obs_frame = env.render()
            done = terminated or truncated

            action = np.zeros(a_dim)
            action[action_scalar] = 1

            next_obs = preprocess_frame(next_obs_frame, img_size, n_channels)

            oar_tuple = [next_obs, action, reward, done]
            agent.replay_buffer.add(oar_tuple)

            obs = next_obs
            ep_steps += 1


    # epsilon schedule
    epsilon_schedule = np.ones(num_episodes) * explore_minLimit
    epsilon_schedule[:int(num_episodes * 0.8)] = np.linspace(explore_maxLimit, explore_minLimit, int(num_episodes * 0.8))


    # interactive episodes
    total_gradient_steps = 0
    interaction_steps = 0 

    for ep in tqdm(range(num_episodes)):
        done = False
        ep_return = 0
        ep_steps = 0
        frames = []

        # first observation of the episode
        observation, info = env.reset()
        obs_frame = env.render()

        # init state and action
        prev_belief = torch.zeros(1, belief_dim)
        prev_state = torch.zeros(1, s_dim)
        prev_action = torch.zeros(1, a_dim)

        # first step 
        action_numpy = np.zeros(a_dim)
        reward = 0
        observation = preprocess_frame(obs_frame, img_size, n_channels)
        oar_tuple = [observation, action_numpy, reward, done]
        agent.replay_buffer.add(oar_tuple)


        agent.rssm.eval()
        with torch.no_grad():
            while not done:

                # infer state from observation using encoder
                prev_state, prev_belief, prev_action = prev_state.to(device), prev_belief.to(device), prev_action.to(device)
                obs_input = torch.FloatTensor(observation).unsqueeze(0).to(device)
                obs_input = obs_input.permute(0,3,1,2) # [1, 3, 210, 160]
                belief, pred_state, enc_state = agent.rssm(prev_belief, prev_state, prev_action, obs_input)

                if ep_steps % action_repeat == 0: # action repeat
                    # sample action from the (stochastic) policy
                    action, policy = agent.get_action(enc_state, belief)
                    # eps-greedy exploration
                    epsilon = epsilon_schedule[ep]
                    action = agent.action_exploration(action, epsilon)
                else:
                    action = prev_action

                action_numpy = action.squeeze(0).detach().cpu().numpy()
                action_scalar = np.argmax(action_numpy).squeeze()
                next_observation, reward, terminated, truncated, _ = env.step(action_scalar)
                next_obs_frame = env.render()
                done = terminated or truncated

                next_observation = preprocess_frame(next_obs_frame, img_size, n_channels)

                oar_tuple = [next_observation, action_numpy, reward, done]
                agent.replay_buffer.add(oar_tuple)

                # record frame (both real and imagination)
                if (ep+1) % record_episode == 0: 
                    # get imagination frame
                    imagined_frame = agent.observation_model.sample(enc_state, belief).squeeze().cpu().detach() 
                    imagined_frame = imagined_frame * 0.5 + 0.5
                    imagined_frame = (imagined_frame * 255).clip(0, 255).int()
                    real_frame = env.render()
                    if n_channels == 1:
                        real_frame = cv2.cvtColor(real_frame, cv2.COLOR_RGB2GRAY) 
                    real_frame = cv2.resize(real_frame, (img_size, img_size), interpolation=cv2.INTER_AREA) 
                    real_frame = torch.tensor(real_frame).clip(0, 255).int()
                    # concat real and imagination frame
                    concat_frame = torch.cat([real_frame, imagined_frame], dim=1)
                    concat_frame = concat_frame.numpy().astype(np.uint8)
                    frames.append(concat_frame)

                # for next step in episode
                observation = next_observation
                prev_belief = belief.detach().cpu()
                prev_state = enc_state.detach().cpu()
                prev_action = action.detach().cpu()

                # ep_return += (df ** ep_steps) * reward
                ep_return += reward
                ep_steps += 1
                interaction_steps += 1

        agent.rssm.train()


        ## episode ended
        # train agent
        if ep % train_episode == 0:
            for _ in range(num_train_calls):

                l_dyn, l_pred, l_sigreg, l_rew, l_obs, l_df, l_act, l_cri, p_entr, ret_scale = agent.train()
                loss_dynamics_list.append(l_dyn.item())
                loss_pred_list.append(l_pred.item())
                loss_sigreg_list.append(l_sigreg.item())
                loss_reward_list.append(l_rew.item())
                loss_obs_list.append(l_obs.item())
                loss_df_list.append(l_df.item())
                loss_actor_list.append(l_act.item())
                loss_critic_list.append(l_cri.item())
                policy_entropy_list.append(p_entr.item())
                ret_scale_list.append(ret_scale.item())

                total_gradient_steps += 1
                if total_gradient_steps % target_critic_update_step == 0:
                    # update critic
                    with torch.no_grad():
                        for target_param, current_param in zip(agent.target_critic_V.parameters(), agent.critic_V.parameters()):
                            target_param.data.copy_(agent.tau * current_param.data + (1 - agent.tau) * target_param.data)

        # store episode stats
        ep_return_list.append(ep_return)
        if ep % (num_episodes//20) == 0:
            print('ep:{} \t ep_return:{}'.format(ep, ep_return))

        # save episode recording 
        if (ep+1) % record_episode == 0:
            imageio.mimsave('../results/' + hyperstr + '_' + str(ep) + '.gif', frames, fps=30)

    print('total interaction steps: ', interaction_steps)



# get moving mean lists
def get_moving_mean_list(a):
    mmlist = [a[0]]
    n = 0
    st = len(a)
    for i in range(1, st):
        n += 1
        n = n % (st//20)
        prev_mean = mmlist[-1]
        new_mean = prev_mean + ((a[i] - prev_mean)/(n+1))
        mmlist.append(new_mean)
    return mmlist

# ep_returns_moving_mean = get_moving_mean_list(ep_return_list)
ep_returns_moving_mean = ep_return_list
loss_dynamics_moving_mean = get_moving_mean_list(loss_dynamics_list)
loss_pred_moving_mean = get_moving_mean_list(loss_pred_list)
loss_sigreg_moving_mean = get_moving_mean_list(loss_sigreg_list)
loss_reward_moving_mean = get_moving_mean_list(loss_reward_list)
loss_obs_moving_mean = get_moving_mean_list(loss_obs_list)
loss_df_moving_mean = get_moving_mean_list(loss_df_list)
loss_actor_moving_mean = get_moving_mean_list(loss_actor_list)
loss_critic_moving_mean = get_moving_mean_list(loss_critic_list)
policy_entropy_moving_mean = get_moving_mean_list(policy_entropy_list)

# plot results
fig, ax = plt.subplots(2,4, figsize=(20,10))

ax[0,0].plot(ep_returns_moving_mean, color='green', label='ep_return')
ax[0,0].legend()
ax[0,0].set_title('return:{:.2f}'.format(ep_returns_moving_mean[-1]))
ax[0,0].set(xlabel='episode')
ax[0,0].grid()

ax[0,1].plot(loss_dynamics_moving_mean, color='blue', label='dynamics_loss')
ax[0,1].plot(loss_pred_moving_mean, color='red', label='pred_loss')
ax[0,1].plot(loss_sigreg_moving_mean, color='green', label='sigreg_loss')
ax[0,1].legend()
ax[0,1].set_title('dyn:{:.2f} pred:{:.2f} sigreg:{:.2f}'.format(loss_dynamics_moving_mean[-1], loss_pred_moving_mean[-1], loss_sigreg_moving_mean[-1]))
ax[0,1].set(xlabel='steps')
ax[0,1].grid()

ax[1,0].plot(loss_actor_moving_mean, color='blue', label='actor_loss')
ax[1,0].legend()
ax[1,0].set_title('actor_loss:{:.3f}'.format(loss_actor_moving_mean[-1]))
ax[1,0].set(xlabel='steps')
ax[1,0].grid()

ax[1,1].plot(loss_critic_moving_mean, color='gray', label='critic_loss')
ax[1,1].legend()
ax[1,1].set_title('critic_loss:{:.3f}'.format(loss_critic_moving_mean[-1]))
ax[1,1].set(xlabel='steps')
ax[1,1].grid()

ax[0,2].plot(loss_reward_moving_mean, color='green', label='reward_loss')
ax[0,2].legend()
ax[0,2].set_title('rew_loss:{:.3f}'.format(loss_reward_moving_mean[-1]))
ax[0,2].set(xlabel='steps')
ax[0,2].grid()

ax[0,3].plot(loss_df_moving_mean, color='magenta', label='df_loss')
ax[0,3].legend()
ax[0,3].set_title('df_loss:{:.3f}'.format(loss_df_moving_mean[-1]))
ax[0,3].set(xlabel='steps')
ax[0,3].grid()

ax[1,2].plot(loss_pred_moving_mean, color='red', label='pred_loss')
ax[1,2].legend()
ax[1,2].set_title(f'pred_loss:{loss_pred_moving_mean[-1]:.3f}') 
ax[1,2].set(xlabel='steps')
ax[1,2].grid()

ax[1,3].plot(policy_entropy_moving_mean, color='blue', label='policy_entropy')
ax[1,3].plot(ret_scale_list, color='green', label='ret_scale')
ax[1,3].legend()
ax[1,3].set_title('policy_entropy:{:.3f} ret_scale:{:2f}'.format(policy_entropy_moving_mean[-1], ret_scale_list[-1]))
ax[1,3].set(xlabel='steps')
ax[1,3].grid()

plt.suptitle('game: ' + game)
plt.savefig('../results/' + hyperstr + '.png')
