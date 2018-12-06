import argparse
from bc_net import BCNet
from data_class import Data
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0,'..')
import load_policy
import tf_util

RANDOM_SEED = 42
MAX_STEPS = 1000

def main(args):
    print('expert:', args['expert'])
    expert_data_path = os.path.join(os.pardir, 'behavioral_cloning', 'expert_data', args['expert'] + '_expert.pkl')
    data = Data(expert_data_path, train_ratio=args['train_ratio'], batch_size=args['batch_size'], only_final_results=args['only_final_results'])
    expert_stats = data.expert_stats
    train_data_loader, val_data_loader = data.get_train_val()
    steps_list, returns_list = run_dagger(args['expert'], args['dagger_iters'], data, args['epochs'], args['only_final_results'])

    if (args['show_plots']):
        show_plots(steps_list, returns_list)

    print('max steps: {}; in iteration {}'.format(np.max(np.asarray(steps_list)), np.argmax(np.asarray(steps_list))))
    print('max return: {:.4f}, in iteration {}'.format(np.max(np.asarray(returns_list)), np.argmax(np.asarray(returns_list))))
    print('mean of returns: {:.4f}'.format(np.mean(np.asarray(returns_list))))
    print('std of returns: {:.4f}'.format(np.std(np.asarray(returns_list))))
    print('mean of returns of expert policy: {:.4f}'.format(expert_stats['mean']))

def run_dagger(expert, dagger_iters, data, epochs, only_final_results):
    expert_policy = load_policy.load_policy(os.path.join(os.pardir, 'experts', expert + '.pkl'))
    steps_list = []
    returns_list = []

    for i in range(dagger_iters):
        print('dagger iter:', i)
        gym.logger.set_level(40)

        all_obs = None
        all_actions = None
        bc_model = BCNet(data.input_dim, data.output_dim)
        optimizer = optim.Adam(bc_model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        train_data_loader, val_data_loader = data.get_train_val()

        train_losses, val_losses = train(epochs, bc_model, optimizer, loss_fn, train_data_loader, val_data_loader, args['only_final_results'])

        with tf.Session():
            tf_util.initialize()

            env = gym.make(expert)
            env.seed(RANDOM_SEED)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = bc_model(torch.Tensor(obs)).detach().numpy()
                obs, r, done, _ = env.step(action)
                expert_action = expert_policy(obs[None,:])
                if all_obs is None:
                    all_obs = obs
                    all_actions = expert_action
                else:
                    all_obs = np.vstack([all_obs, obs])
                    all_actions = np.vstack([all_actions, expert_action])
                totalr += r
                steps += 1

                if  (not only_final_results) and (steps % 100 == 0): 
                    print("%i/%i"%(steps, MAX_STEPS))
                if steps >= MAX_STEPS:
                    done = True

            print('Number of steps in this rollout: {}'.format(steps))
            steps_list.append(steps)
            print('total return: {:.3f}'.format(totalr))
            returns_list.append(totalr)

            all_obs = np.vstack([all_obs, data.X_train, data.X_val])
            all_actions = np.vstack([all_actions, data.y_train, data.y_val])
            in_data = {'observations' : all_obs, 
                       'actions' : all_actions, 
                       'expert_stats' : data.expert_stats}
            data = Data(in_data)
            train_data_loader, val_data_loader = data.get_train_val()

    return steps_list, returns_list

def train(epochs, network, optimizer, loss_fn, train_loader, test_loader, only_final_results, testing=True):
    train_losses = []
    val_losses = []
    training_step = 0
    network.train()
    for epoch in range(1, epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = loss_fn(output, target)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            training_step += 1

            if (not only_final_results) and ((training_step % 40 == 0) or (training_step == 1)) :
                print('Epoch {}; training step {} mse: {:.3f}'.format(epoch, training_step, loss.item()))
            
        if (testing):
            val_losses.append(validate(network, test_loader, loss_fn, only_final_results))   
            
    return train_losses, val_losses

def validate(network, test_loader, loss_fn, only_final_results):
    loss = 0
    steps = 0
    network.eval()
    
    for batch_idx, (data, target) in enumerate(test_loader):
        output = network(data)
        loss += loss_fn(output, target).item()
        steps += 1
        
    avg_loss = loss / steps
    if not only_final_results:
        print('Test set mse: {:.3f}'.format(avg_loss))    
    return avg_loss

def show_plots(steps_list, returns_list):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(np.asarray(returns_list))
    plt.ylabel('returns')
    plt.xlabel('dagger iteration')
    plt.subplot(212)
    plt.plot(np.asarray(steps_list))
    plt.ylabel('steps')
    plt.xlabel('dagger iteration')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert', type=str)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--dagger_iters', type=int, default=200)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--only_final_results', type=bool, default=False)
    parser.add_argument('--show_plots', type=bool, default=False)
    args = vars(parser.parse_args())
    main(args)
