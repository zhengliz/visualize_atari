# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License
# This file contains a series of bash commands for generating saliency videos that might be of interest

# NOTE: the pretrained agents we provide in this repo are _slightly_ different from the ones used to obtain results in the paper (our code has evolved a great deal over the course of the past several months). Same hyperparameters and optimization...but different implementations. We made sure that these agents work, but you may need to fine-tune a few of the saliency parameters to get the best results.

# strong agent videos
python3 make_movie.py --prefix strong --checkpoint strong.40.tar --first_frame 350 --env Pong-v0
python3 make_movie.py --prefix strong --checkpoint strong.40.tar --first_frame 350 --env SpaceInvaders-v0
python3 make_movie.py --prefix strong --checkpoint strong.40.tar --first_frame 350 --env Breakout-v0

# overfit agent videos
python3 make_movie.py --prefix overfit --checkpoint overfit.3.tar --first_frame 350 --env Pong-v0 --overfit_mode True
python3 make_movie.py --prefix control --checkpoint control.3.tar --first_frame 350 --env Pong-v0 --overfit_mode True

python3 make_movie.py --prefix overfit --checkpoint overfit.3.tar --first_frame 350 --env SpaceInvaders-v0 --overfit_mode True
python3 make_movie.py --prefix control --checkpoint control.3.tar --first_frame 350 --env SpaceInvaders-v0 --overfit_mode True

python3 make_movie.py --prefix overfit --checkpoint overfit.3.tar --first_frame 350 --env Breakout-v0 --overfit_mode True
python3 make_movie.py --prefix control --checkpoint control.3.tar --first_frame 350 --env Breakout-v0 --overfit_mode True

# learning agent videos
# We are still cleaning up the code for generating the learning visualizations. Doing it yourself is not difficult; simply run the same code you would for a strong agent, and then load your partially-trained agents and pass them through deterministic rollouts where the states/actions are taken from the rollout of the strong agent. If you have details/questions about how to do this, email Sam at: greydanus (dot) 17 (at) gmail (dot) com.


#python3 make_movie.py --prefix strong --checkpoint strong.40.tar --first_frame 350 --env Pong-v0 --delta 0
#python3 make_movie.py --prefix strong --checkpoint strong.40.tar --first_frame 350 --env SpaceInvaders-v0 --delta 0
#python3 make_movie.py --prefix strong --checkpoint strong.40.tar --first_frame 350 --env Breakout-v0 --delta 0
#python3 make_movie.py --prefix overfit --checkpoint overfit.3.tar --first_frame 350 --env Pong-v0 --overfit_mode True --delta 0
#python3 make_movie.py --prefix control --checkpoint control.3.tar --first_frame 350 --env Pong-v0 --overfit_mode True --delta 0
#python3 make_movie.py --prefix overfit --checkpoint overfit.3.tar --first_frame 350 --env SpaceInvaders-v0 --overfit_mode True --delta 0
#python3 make_movie.py --prefix control --checkpoint control.3.tar --first_frame 350 --env SpaceInvaders-v0 --overfit_mode True --delta 0
#python3 make_movie.py --prefix overfit --checkpoint overfit.3.tar --first_frame 350 --env Breakout-v0 --overfit_mode True --delta 0
#python3 make_movie.py --prefix control --checkpoint control.3.tar --first_frame 350 --env Breakout-v0 --overfit_mode True --delta 0