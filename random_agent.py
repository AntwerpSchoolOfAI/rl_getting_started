import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from env_wrappers import GreyscaleObservationWrapper, ResizeObservationWrapper, MaxAndSkipEnv, FrameStack, NoopResetEnv

class DQNModel(nn.Module):

    def __init__(self, actions):

        super(DQNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc4 = nn.Linear(38 * 7 * 64, 512) 
        self.fc5 = nn.Linear(512, actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

# example of how to train your network with new updated q-estimates
def train_network(q_online, q_target):

    # Compute Huber loss
    loss = F.smooth_l1_loss(q_online, q_target.detach())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)


# prepare env and do pre-processing
env = gym.make('BreakoutNoFrameskip-v4')
env = ResizeObservationWrapper(env, width=84, height=84)
env = GreyscaleObservationWrapper(env)
env = MaxAndSkipEnv(env)
env = FrameStack(env, 4)
env = NoopResetEnv(env)

state = env.reset()

model = DQNModel(actions=env.action_space.n)
optimizer = torch.optim.RMSprop(model.parameters())

for _ in range(1000):
    env.render()

    # use your network:
    state = np.array(state)
    transformed_state = np.reshape(state, (1, 1, state.shape[1], state.shape[2]))
    transformed_state_tensor = torch.tensor(transformed_state.tolist())
    q_values = model(transformed_state_tensor)
    print(q_values)

    # TODO: also take greedy actions
    action = env.action_space.sample() # take a random action

    state, reward, done, _ = env.step(action)

    # update network
    q_new = q_values.clone()
    # TODO: update q_new values (also use exp replay)
    train_network(q_new, q_values)

    if done:
        env.reset()
