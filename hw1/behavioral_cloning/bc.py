import argparse
from bc_net import BCNet
from data_class import Data
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

def main(args):
    print('expert:', args['expert'])
    expert_data_path = os.path.join(os.getcwd(), 'expert_data', args['expert'] + '_expert.pkl')
    data = Data(expert_data_path, train_ratio=args['train_ratio'], batch_size=args['batch_size'])
    train_data_loader, val_data_loader = data.get_train_val()
    bc_model = BCNet(data.input_dim, data.output_dim)
    optimizer = config_optimizer(args['optimizer'], args['lr'], bc_model)
    train_losses, val_losses = train(args['epochs'], 
                                        bc_model, 
                                        optimizer, 
                                        nn.MSELoss(), 
                                        train_data_loader, 
                                        val_data_loader)
    if args['show_plots']:
        show_plots(train_losses, val_losses)

    test_with_gym(bc_model, args['rollouts'], args['render'], data.expert_stats)

def config_optimizer(optimizer, lr, model):
    if optimizer=='adam':
        return optim.Adam(model.parameters(), lr=lr)
    else:
        print('no valid keywork found for optimizer; setting to SGD')
        return optim.SGD(model.parameters(), lr=lr)

def train(epochs, network, optimizer, loss_fn, train_loader, test_loader, testing=True):
    assert (epochs >= 1), 'epochs must be a positive integer with value at least 1'
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

            if (training_step % 40 == 0) or (training_step == 1) :
                print('Epoch {}; training step {} mse: {:.3f}'.format(epoch, training_step, loss.item()))
            
        if (testing):
            val_losses.append(validate(network, test_loader, loss_fn))   
            
    return train_losses, val_losses

def validate(network, test_loader, loss_fn):
    loss = 0
    steps = 0
    network.eval()
    
    for batch_idx, (data, target) in enumerate(test_loader):
        output = network(data)
        loss += loss_fn(output, target).item()
        steps += 1
        
    avg_loss = loss / steps
    print('Test set mse: {:.3f}'.format(avg_loss))    
    return avg_loss

def show_plots(train_losses, val_losses):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(train_losses)
    plt.ylabel('training error')
    plt.xlabel('training step')
    plt.subplot(212)
    plt.plot(val_losses)
    plt.ylabel('validation error')
    plt.xlabel('epoch')
    plt.show()

def test_with_gym(policy_fn, rollouts, render, expert_stats):
    gym.logger.set_level(40)
    random_seed=42
    env = gym.make('Hopper-v2')
    env.seed(random_seed)
    max_steps = 1000
    returns = []
    
    for i in range(rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        
        while not done:
            action = policy_fn(torch.Tensor(obs)).detach().numpy()
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            
            if render:
                env.render()
            if steps % 100 == 0: 
                print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        print('Number of steps in this rollout: {}'.format(steps))
        returns.append(totalr)
        
    print('bc_model: mean of return', np.mean(returns))
    print('bc_model: std of return', np.std(returns))
    print('expert: mean of return', expert_stats['mean'])
    print('expert: std of return', expert_stats['std'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert', type=str)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--show_plots', type=bool, default=True)
    parser.add_argument('--rollouts', type=int, default=10)
    parser.add_argument('--render', type=bool, default=False)
    args = vars(parser.parse_args())
    main(args)

