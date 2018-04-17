from __future__ import print_function
import warnings
warnings.filterwarnings('ignore') # mute warnings, live dangerously
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.animation as manimation
# manimation.verbose.set_level('debug')

import gym, os, sys, time, argparse

sys.path.append('..')
from visualize_atari import *


def make_movie(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # set up dir variables and environment
    load_dir = '{}{}/'.format('overfit-' if args.overfit_mode else '', args.env_name.lower())
    meta = get_env_meta(args.env_name)

    if not args.overfit_mode:
        env = gym.make(args.env_name).unwrapped
    else:
        env = OverfitAtari(args.env_name, load_dir+'expert/', seed=args.seed) # make a seeded env
    env.seed(args.seed)
    env.reset()

    # set up agent
    model = NNPolicy(channels=1, num_actions=env.action_space.n)
    model.try_load(load_dir, checkpoint=args.checkpoint)

    # get a rollout of the policy
    movie_title = "{}-{}-{}.mp4".format(args.prefix, args.num_frames, args.env_name.lower())
    print('\tmaking movie "{}" using checkpoint at {}{}'.format(movie_title, load_dir, args.checkpoint))
    max_ep_len = args.first_frame + args.num_frames + 1

    if args.delta is None:
        history = rollout(model, env, max_ep_len=max_ep_len, store_env=False)
    else:
        history = rollout(model, env, max_ep_len=max_ep_len, store_env=True)
    print()

    # make the movie!
    start = time.time()
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=movie_title, artist='greydanus', comment='atari-saliency-video')
    writer = FFMpegWriter(fps=8, metadata=metadata)

    prog = '' ; total_frames = len(history['ins'])
    max_frames = total_frames - 1 if args.delta is None else total_frames - args.delta - 1
    f = plt.figure(figsize=[6, 6*1.3], dpi=args.resolution)
    with writer.saving(f, args.save_dir + movie_title, args.resolution):
        for i in range(args.num_frames):
            ix = args.first_frame + i
            if ix < max_frames: # prevent loop from trying to process a frame ix greater than rollout length
                frame = history['ins'][ix].squeeze().copy()
                actor_saliency = score_frame(model, history, ix, env, args.radius, args.density, interp_func=occlude, mode='actor', delta=args.delta)
                critic_saliency = score_frame(model, history, ix, env, args.radius, args.density, interp_func=occlude, mode='critic', delta=args.delta)
            
                frame = saliency_on_atari_frame(actor_saliency, frame, fudge_factor=meta['actor_ff'], channel=2)
                frame = saliency_on_atari_frame(critic_saliency, frame, fudge_factor=meta['critic_ff'], channel=0)

                plt.imshow(frame) ; plt.title(args.env_name.lower(), fontsize=15)
                writer.grab_frame() ; f.clear()
                
                tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
                print('\ttime: {} | progress: {:.1f}%'.format(tstr, 100*i/min(args.num_frames, total_frames)), end='\r')
    print('\nfinished.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_name', default='Breakout-v0', type=str, help='gym environment')
    parser.add_argument('-d', '--density', default=5, type=int, help='density of grid of gaussian blurs')
    parser.add_argument('-r', '--radius', default=5, type=int, help='radius of gaussian blur')
    parser.add_argument('-f', '--num_frames', default=20, type=int, help='number of frames in movie')
    parser.add_argument('-i', '--first_frame', default=150, type=int, help='index of first frame')
    parser.add_argument('-dpi', '--resolution', default=75, type=int, help='resolution (dpi)')
    parser.add_argument('-s', '--save_dir', default='./movies/', type=str, help='dir to save agent logs and checkpoints')
    parser.add_argument('-p', '--prefix', default='default', type=str, help='prefix to help make video name unique')
    parser.add_argument('-c', '--checkpoint', default='*.tar', type=str, help='checkpoint name (in case there is more than one')
    parser.add_argument('-o', '--overfit_mode', default=False, type=bool, help='analyze an overfit environment (see paper)')
    parser.add_argument('--seed', default=326, type=int, help='random seed')
    parser.add_argument('--delta', default=None, type=int, help='delta t advanced in time')

    args = parser.parse_args()

    if args.delta is not None:
        args.save_dir = args.save_dir.rstrip('/') + '_delta{}/'.format(args.delta)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    make_movie(args)
