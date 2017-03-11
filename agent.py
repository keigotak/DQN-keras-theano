import os
import datetime
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from collections import deque
import random
from keras import backend as K


class Agent:

    def __init__(self, memory_size=1000, nb_frames=None, grid_size=80, nb_action=0):
        self.nb_frames = nb_frames
        self.grid_size = grid_size
        self.nb_action = nb_action
        self.s, self.q_values, self.q_network = self.build_network()
        self.st, self.target_q_values, self.target_network = self.build_network()
        self.replay_memory = deque()
        assert len(self.q_network.output_shape) == 2, "Model's output shape should be (nb_samples, nb_actions)."
        if not nb_frames and not self.q_network.input_shape:
            raise Exception("Missing argument : nb_frames not provided")
        elif self.q_network.input_shape[1] and nb_frames and self.q_network.input_shape[1] != nb_frames:
            raise Exception("Dimension mismatch : time dimension of model should be equal to nb_frames.")

        self.memory_size = memory_size
        self.epsilon_flg = False
        self.epsilon = 1.0
        self.final_epsilon = 0.0
        self.delta_epsilon = 0.0
        self.action_interval = 4
        self.train_interval = 4
        self.initial_replay_size = memory_size / 20
        self.no_op_steps = 30
        self.weight_update_freq = 10000
        self.frames = None
        self.save_name = "bl"
        self.save_name_index = ""
        self.save_dir = "weights"
        self.save_freq = 50
        self.frame_num = 0
        self.lname = "log_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".csv"
        f = open(os.path.join(self.save_dir, self.lname), "a")
        f.write("Epoch,Loss,Epsilon,Win,Earned\n")
        f.close()
        self.lname_p = "log_play_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".csv"
        f = open(os.path.join(self.save_dir, self.lname_p), "a")
        f.write("Epoch,Acc,Epsilon,Win,Earned\n")
        f.close()

        try:
            self.q_network.load_weights('{}.h5'.format(self.save_name))
            self.target_network.load_weights('{}.h5'.format(self.save_name))
            print "loading from {}.h5".format(os.path.join(self.save_dir, self.save_name + self.save_name_index))
        except:
            print "Training a new model"

    def build_network(self):
        model = Sequential()
        model.add(BatchNormalization(axis=1, input_shape=(self.nb_frames, self.grid_size, self.grid_size), mode=2))
        model.add(Convolution2D(32, nb_row=8, nb_col=8, subsample=(4, 4), border_mode='same', activation='relu'))
        model.add(Convolution2D(64, nb_row=4, nb_col=4, subsample=(2, 2), border_mode='same', activation='relu'))
        model.add(Convolution2D(64, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.nb_action))
        model.compile(RMSprop(), loss='mse')

        s = K.placeholder(shape=(None, self.nb_frames, self.grid_size, self.grid_size))
        q_values = model(s)

        return s, q_values, model

    @property
    def memory_size(self):
        return len(self.replay_memory)

    def reset_memory(self):
        self.replay_memory.clear()

    def check_game_compatibility(self, game):
        game_output_shape = (1, None) + game.get_frame().shape
        if len(game_output_shape) != len(self.q_network.input_shape):
            raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
        else:
            for i in range(len(self.q_network.input_shape)):
                if self.q_network.input_shape[i] and game_output_shape[i] and self.q_network.input_shape[i] != game_output_shape[i]:
                    raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
        if len(self.q_network.output_shape) != 2 or self.q_network.output_shape[1] != game.nb_actions:
            raise Exception('Output shape of model should be (nb_samples, nb_actions).')

    def get_game_data(self, game):
        frame = game.get_frame()
        if self.frames is None:
            self.frames = [frame] * self.nb_frames
        else:
            self.frames.append(frame)
            self.frames.pop(0)
        return np.expand_dims(self.frames, 0)

    def clear_frames(self):
        self.frames = None

    def update_weights(self):
        weights = self.q_network.get_weights()
        self.target_network.set_weights(weights)

    @staticmethod
    def loss_teacher_signal(y_true, y_pred):
        error = K.abs(y_true - y_pred)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        return loss

    def train(self, game, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon=[1., .1], epsilon_rate=100, reset_memory=False):
        self.check_game_compatibility(game)
        if not self.epsilon_flg:
            if type(epsilon) in {tuple, list}:
                self.final_epsilon = epsilon[1]
                self.epsilon = epsilon[0]
                self.delta_epsilon = ((self.epsilon - self.final_epsilon) / (nb_epoch * epsilon_rate))
            else:
                self.delta_epsilon = 0
                self.final_epsilon = self.epsilon
        self.epsilon_flg = True
        win_count = 0
        for epoch in range(nb_epoch):
            loss = 0.
            frame = 1
            game.reset()
            self.clear_frames()
            if reset_memory:
                self.reset_memory()
            game_over = False
            observation = self.get_game_data(game)
            next_observation = observation
            action = 2
            for _ in xrange(np.random.randint(1, self.no_op_steps)):
                observation = next_observation
                game.play(action)
                next_observation = self.get_game_data(game)

            action = int(np.random.randint(game.nb_actions))
            while not game_over:
                if self.frame_num % self.action_interval == 0:
                    if np.random.random() < self.epsilon or self.frame_num < self.initial_replay_size:
                        action = int(np.random.randint(game.nb_actions))
                    else:
                        q_value = self.q_network.predict(observation)
                        action = int(np.argmax(q_value[0]))
                game.play(action)
                reward = np.sign(game.get_score())
                next_observation = self.get_game_data(game)
                game_over = game.is_over()
                self.replay_memory.append((observation, action, reward, next_observation, game_over))
                if len(self.replay_memory) > self.memory_size:
                    self.replay_memory.popleft()
                observation = next_observation

                if self.frame_num > self.initial_replay_size:
                    if self.frame_num % self.train_interval == 0:
                        # batch shape should be None, 4, 80, 80
                        minibatch = random.sample(self.replay_memory, batch_size)
                        state_batch = []
                        action_batch = []
                        reward_batch = []
                        next_state_batch = []
                        terminal_batch = []
                        for data in minibatch:
                            state_batch.append(data[0])
                            action_batch.append(data[1])
                            reward_batch.append(data[2])
                            next_state_batch.append(data[3])
                            terminal_batch.append(data[4])
                        terminal_batch = np.array(terminal_batch) + 0
                        npa_next_state_batch = np.float32(np.array(next_state_batch))
                        npa_next_state_batch = np.reshape(npa_next_state_batch, (batch_size, self.nb_frames, self.grid_size, self.grid_size))
                        npa_state_batch = np.float32(np.array(state_batch))
                        npa_state_batch = np.reshape(npa_state_batch, (batch_size, self.nb_frames, self.grid_size, self.grid_size))

                        target_q_values_batch = self.target_network.predict(npa_next_state_batch)
                        y_batch = reward_batch + (1 - terminal_batch) * gamma * np.max(target_q_values_batch, axis=1)

                        q_values_batch = self.q_network.predict(npa_next_state_batch)
                        for i in xrange(len(q_values_batch)):
                            if terminal_batch[i]:
                                q_values_batch[i, action_batch[i]] = reward_batch[i]
                            else:
                                q_values_batch[i, action_batch[i]] = y_batch[i]

                        loss += float(self.q_network.train_on_batch(npa_state_batch, q_values_batch))

                # print("Epoch {:03d} | Frame score {} | Total score {} | Action {} | Earned {}".format(epoch + 1, reward, game.score, action, game.get_earned()))
                self.frame_num += 1
                if self.epsilon > self.final_epsilon and self.frame_num >= self.initial_replay_size:
                    self.epsilon -= self.delta_epsilon
                elif self.epsilon < self.final_epsilon:
                    self.epsilon = self.final_epsilon

                if self.frame_num % self.weight_update_freq == 0:
                    self.update_weights()

                frame += 1

            if game.is_won():
                win_count += 1
            loss = loss / frame

            print("Epoch {:03d}/{:03d} | Loss {:.6f} | Epsilon {:.7f} | Win {} | Earned {}".format(epoch + 1, nb_epoch, loss, self.epsilon, win_count, game.get_earned()))
            f = open(os.path.join(self.save_dir, self.lname), "a")
            f.write("{:03d},{:.4f},{:.7f},{},{}\n".format(epoch + 1, loss, self.epsilon, win_count, game.get_earned()))
            f.close()

            if epoch % self.save_freq == 0:
                print "Save weight file of epoch:", epoch
                self.q_network.save_weights('{}.h5'.format(os.path.join(self.save_dir, self.save_name + "_" + str(epoch))), True)
                self.q_network.save_weights('{}.h5'.format(os.path.join(self.save_dir, self.save_name)), True)

    def play(self, game, nb_epoch=10, epsilon=0.):
        self.check_game_compatibility(game)
        q_network = self.q_network
        win_count = 0
        for epoch in range(nb_epoch):
            game.reset()
            self.clear_frames()
            observation = self.get_game_data(game)
            game_over = False
            while not game_over:
                if np.random.rand() < epsilon:
                    print("random")
                    action = int(np.random.randint(0, game.nb_actions))
                else:
                    q_value = q_network.predict(observation)
                    action = int(np.argmax(q_value[0]))
                game.play(action)
                # print("Epoch {:03d} | Frame score {} | Total score {} | Action {} | Earned {}".format(epoch + 1, game.get_score(), game.score, action, game.get_earned()))
                observation = self.get_game_data(game)
                game_over = game.is_over()
            if game.is_won():
                win_count += 1
            acc = 100. * win_count / (epoch + 1)
            print("Epoch {:03d}/{:03d} | Acc {:.6f} | Epsilon {:.7f} | Win {} | Earned {}".format(epoch + 1, nb_epoch, acc, epsilon, win_count, game.get_earned()))
            f = open(os.path.join(self.save_dir, self.lname_p), "a")
            f.write("{:03d},{:.4f},{:.7f},{},{}\n".format(epoch + 1, acc, epsilon, win_count, game.get_earned()))
            f.close()
        print("Accuracy {} %".format(100. * win_count / nb_epoch))
