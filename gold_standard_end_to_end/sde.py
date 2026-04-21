"""SDE utilities for training and sampling polygon alignment actions."""

import functools

import torch

SIGMA_MIN_VEC = torch.tensor([1.0, 1.0, 1.0, 1.0])
SIGMA_MAX_VEC = torch.tensor([250.0, 250.0, 50.0, 50.0])


def ve_marginal_prob(x, t, sigma_min=SIGMA_MIN_VEC, sigma_max=SIGMA_MAX_VEC):
    """Variance exploding marginal with per-dimension noise scales."""
    sigma_min = sigma_min.to(x.device)
    sigma_max = sigma_max.to(x.device)
    t_expand = t.view(-1, 1, 1)
    std = sigma_min * (sigma_max / sigma_min).pow(t_expand)
    return x, std


def ve_sde(t, sigma_min=SIGMA_MIN_VEC, sigma_max=SIGMA_MAX_VEC):
    """Return drift and diffusion coefficients for the VE SDE."""
    sigma_min = sigma_min.to(t.device)
    sigma_max = sigma_max.to(t.device)
    t_expand = t.view(-1, 1, 1)
    sigma_t = sigma_min * (sigma_max / sigma_min).pow(t_expand)
    log_term = torch.log(sigma_max / sigma_min)
    diffusion = sigma_t * torch.sqrt(2 * log_term).view(1, 1, -1)
    drift = torch.zeros_like(diffusion)
    return drift, diffusion


def ve_prior(shape, sigma_max=SIGMA_MAX_VEC):
    """Sample the VE prior x_1 ~ Normal(0, diag(sigma_max^2))."""
    sigma_max = sigma_max.to(shape[0] if isinstance(shape, torch.Tensor) else "cpu")
    noise = torch.randn(*shape, device=sigma_max.device)
    return noise * sigma_max.view(1, 1, 4)


def vp_marginal_prob(x, t, beta_0=0.1, beta_1=20):
    log_mean_coeff = -0.25 * t**2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff) * x
    std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
    return mean, std


def vp_sde(t, beta_0=0.1, beta_1=20):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift_coeff = -0.5 * beta_t
    diffusion_coeff = torch.sqrt(beta_t)
    return drift_coeff, diffusion_coeff


def vp_prior(shape, beta_0=0.1, beta_1=20):
    _ = beta_0, beta_1
    return torch.randn(*shape)


def subvp_marginal_prob(x, t, beta_0, beta_1):
    log_mean_coeff = -0.25 * t**2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff) * x
    std = 1 - torch.exp(2.0 * log_mean_coeff)
    return mean, std


def subvp_sde(t, beta_0, beta_1):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift_coeff = -0.5 * beta_t
    discount = 1.0 - torch.exp(-2 * beta_0 * t - (beta_1 - beta_0) * t**2)
    diffusion_coeff = torch.sqrt(beta_t * discount)
    return drift_coeff, diffusion_coeff


def subvp_prior(shape, beta_0=0.1, beta_1=20):
    _ = beta_0, beta_1
    return torch.randn(*shape)


def init_sde(sde_mode):
    """Build SDE helper functions matching the selected formulation."""
    if sde_mode == "ve":
        eps = 1e-5
        prior_fn = functools.partial(ve_prior)
        marginal_prob_fn = functools.partial(ve_marginal_prob)
        sde_fn = functools.partial(ve_sde)
    elif sde_mode == "vp":
        beta_0 = 0.1
        beta_1 = 20
        eps = 1e-3
        prior_fn = functools.partial(vp_prior, beta_0=beta_0, beta_1=beta_1)
        marginal_prob_fn = functools.partial(vp_marginal_prob, beta_0=beta_0, beta_1=beta_1)
        sde_fn = functools.partial(vp_sde, beta_0=beta_0, beta_1=beta_1)
    elif sde_mode == "subvp":
        beta_0 = 0.1
        beta_1 = 20
        eps = 1e-3
        prior_fn = functools.partial(subvp_prior, beta_0=beta_0, beta_1=beta_1)
        marginal_prob_fn = functools.partial(subvp_marginal_prob, beta_0=beta_0, beta_1=beta_1)
        sde_fn = functools.partial(subvp_sde, beta_0=beta_0, beta_1=beta_1)
    else:
        raise NotImplementedError(f"Unsupported sde_mode: {sde_mode}")
    return prior_fn, marginal_prob_fn, sde_fn, eps


def pc_sampler_state(
    score_model,
    sde_fn,
    g1,
    g2,
    num_steps=256,
    snr=0.16,
    corrector_steps=4,
):
    """Predictor-corrector sampler for action states."""
    device = next(score_model.parameters()).device
    batch_size = g1.num_graphs if hasattr(g1, "num_graphs") else 1

    sigma_max_vec = SIGMA_MAX_VEC.to(device)
    actions = torch.randn(batch_size, 2, 4, device=device) * sigma_max_vec.view(1, 1, 4)

    time_steps = torch.linspace(1.0, 1e-3, num_steps, device=device)
    dt = time_steps[0] - time_steps[1]
    traj = []

    with torch.no_grad():
        for t in time_steps:
            t_batch = torch.full((batch_size,), t, device=device)
            _, g = sde_fn(t_batch)
            score = score_model(g1, g2, actions, t_batch)

            for _ in range(corrector_steps):
                noise = torch.randn_like(actions)
                grad_norm = torch.norm(score.reshape(batch_size, -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(batch_size, -1), dim=-1).mean()
                step_size = (snr * noise_norm / (grad_norm + 1e-6)) ** 2 * 2
                step_size = step_size.view(batch_size, 1, 1)
                actions = actions + step_size * score + torch.sqrt(2 * step_size) * noise

            noise = torch.randn_like(actions)
            drift_rev = (g**2) * score
            actions = actions + drift_rev * dt + g * torch.sqrt(dt) * noise
            traj.append(actions.unsqueeze(1))

    return torch.cat(traj, dim=1), actions.detach().cpu()


def lossFun(model, g1, g2, actions, marginalProbFunc):
    """Score-matching loss for the polygon diffusion model."""
    device = next(model.parameters()).device
    actions = actions.to(device)
    g1 = g1.to(device)
    g2 = g2.to(device)

    batch_size = actions.size(0)
    t = torch.rand(batch_size, device=device) * (1.0 - 0.001) + 0.001
    noise = torch.randn_like(actions, device=device)
    mu, std = marginalProbFunc(actions, t)
    std = std + 1e-5
    actions_t = mu + noise * std
    pred = model(g1, g2, actions_t, t)
    delta = pred * std + noise
    loss = (delta**2).mean()
    return loss, delta.detach()


class ExponentialMovingAverage:
    """Maintain an exponential moving average of model parameters."""

    def __init__(self, parameters, decay, use_num_updates=True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = state_dict["shadow_params"]
