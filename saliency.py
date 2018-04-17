from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.
searchlight = lambda I, mask: I*mask + gaussian_filter(I, sigma=3)*(1-mask) # choose an area NOT to blur
occlude = lambda I, mask: I*(1-mask) + gaussian_filter(I, sigma=3)*mask # choose an area to blur


def get_mask(center, size, r):
    y, x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()


def run_through_model(model, history, ix, interp_func=None, mask=None, mode='actor'):
    if mask is None:
        im = prepro(history['ins'][ix])
    else:
        assert(interp_func is not None, "interp func cannot be none")
        im = interp_func(prepro(history['ins'][ix]).squeeze(), mask).reshape(1, 80, 80) # perturb input I -> I'
    tens_state = torch.FloatTensor(im)
    state = Variable(tens_state.unsqueeze(0), volatile=True)
    hx = Variable(torch.FloatTensor(history['hx'][ix]).view(1, -1))
    cx = Variable(torch.FloatTensor(history['cx'][ix]).view(1, -1))
    return model((state, (hx, cx)))[0] if mode == 'critic' else model((state, (hx, cx)))[1]


def run_through_model_w_delta(model, history, ix, env, interp_func=None, mask=None, mode='actor', delta=2):
    if mask is None:
        if mode == 'critic':
            return Variable(torch.FloatTensor(history['values'][ix+delta+1])).view(1, -1)
        else:
            return Variable(torch.FloatTensor(history['logits'][ix+delta+1])).view(1, -1)

    assert (interp_func is not None, "interp func cannot be none")

    im = interp_func(prepro(history['ins'][ix]).squeeze(), mask).reshape(1, 80, 80)  # perturb input I -> I'
    tens_state = torch.FloatTensor(im)
    state = Variable(tens_state.unsqueeze(0), volatile=True)
    hx = Variable(torch.FloatTensor(history['hx'][ix]).view(1, -1))
    cx = Variable(torch.FloatTensor(history['cx'][ix]).view(1, -1))
    c_l, a_l, (hx, cx) = model((state, (hx, cx)))

    env.restore_full_state(history['env'][ix])
    for i in range(delta):
        action = F.softmax(a_l).max(1)[1].data # F.softmax(a_l).multinomial().data[0]
        obs, reward, done, expert_policy = env.step(action.numpy()[0])
        state = torch.FloatTensor(prepro(obs))
        c_l, a_l, (hx, cx) = model((Variable(state.view(1, 1, 80, 80)), (hx.detach(), cx.detach())))

    return c_l.detach() if mode == 'critic' else a_l.detach()


def score_frame(model, history, ix, env, r, d, interp_func, mode='actor', delta=None):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)
    assert mode in ['actor', 'critic'], 'mode must be either "actor" or "critic"'
    if delta is None:
        L = run_through_model(model, history, ix, interp_func, mask=None, mode=mode)
    else:
        L = run_through_model_w_delta(model, history, ix, env, interp_func, mask=None, mode=mode, delta=delta)
    scores = np.zeros((int(80/d)+1, int(80/d)+1)) # saliency scores S(t,i,j)
    for i in range(0, 80, d):
        for j in range(0, 80, d):
            mask = get_mask(center=[i, j], size=[80, 80], r=r)
            if delta is None:
                l = run_through_model(model, history, ix, interp_func, mask=mask, mode=mode)
            else:
                l = run_through_model_w_delta(model, history, ix, env, interp_func, mask=mask, mode=mode, delta=delta)
            scores[int(i/d), int(j/d)] = (L-l).pow(2).sum().mul_(.5).data[0]
    pmax = scores.max()
    scores = imresize(scores, size=[80, 80], interp='bilinear').astype(np.float32)
    return pmax * scores / scores.max()


def saliency_on_atari_frame(saliency, atari, fudge_factor, channel=2, sigma=0):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    pmax = saliency.max()
    S = imresize(saliency, size=[160,160], interp='bilinear').astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S -= S.min()
    S = fudge_factor*pmax * S / S.max()
    I = atari.astype('uint16')
    I[35:195, :, channel] += S.astype('uint16')
    I = I.clip(1, 255).astype('uint8')
    return I


def get_env_meta(env_name):
    meta = {}
    if env_name == "Pong-v0":
        meta['critic_ff'] = 600
        meta['actor_ff'] = 500
    elif env_name == "Breakout-v0":
        meta['critic_ff'] = 600
        meta['actor_ff'] = 300
    elif env_name == "SpaceInvaders-v0":
        meta['critic_ff'] = 400
        meta['actor_ff'] = 400
    else:
        print('environment "{}" not supported'.format(env_name))
    return meta
