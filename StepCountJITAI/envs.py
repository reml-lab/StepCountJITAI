import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
from collections import deque
import gymnasium as gym
from gym import spaces
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class StepCountJITAI(gym.Env):
  def __init__(self, sigma, chosen_obs_names, chosen_context_for_P=1,
               b_using_uniform=True, chosen_a=0.01,       # chosen_a is used   when using b_using_uniform=True
               n_version=0,  κh=None, κd=None, σs=None,   # κh and κd are used when using b_using_uniform=False
               #n_version=1, κh=10., κd=10., σs=10.,
               #n_version=1, κh=10., κd=10., σs=25.,
               seed=0, max_episode_length=50, n_context=2, n_actions=4,
               δh=0.1, εh=0.05, δd=0.1, εd=0.4, µs=[0.1,0.1], ρ1=50., ρ2=200., ρ3=0., D_threshold=.99,
               b_add_time_inhomogeneous=False, b_display=True, b_gymnasium=True):
    r"""This is the class for StepCountJITAI environment.
    The possible obs names for chosen_obs_names are: 'C', 'P', 'L', 'H', 'D', 'Z', 'W', 'T',
    for example: chosen_obs_names=['C','H','D'] or ['P', 'T'], where C is for true contex, L is for inferred context,
    H is for habituation, D is for disengagement, Z is for new state, W is for another new state, and T is the binary indicator.
    chosen_context_for_P=1 indicates that P is the probability of context 1.
    The default version is n_version = 0 (deterministic). The stochastic version is n_version = 1 (κh, κd and σs can be set).
    b_using_uniform=True indicates that the stochatic version will use Uniform distribution (instead of Beta distribution).
    b_add_time_inhomogeneous=True (default is False) allows to augment obs with a one-hot vector of length max_episode_length, for example: [0,0,0...1,...0,0]."""
    super(StepCountJITAI, self).__init__()
    self.b_gymnasium = b_gymnasium
    self.max_episode_length = max_episode_length
    self.n_version = n_version
    if self.n_version == 1:
      if σs is None:
        assert(1==2), 'error: σs must be set when using the stochastic version (e.g., set σs=10).' 
      self.κh = κh
      self.κd = κd
      self.σs = σs
    self.C = n_context
    self.sigma = sigma
    self.chosen_obs_names = chosen_obs_names
    self.chosen_obs_names_str = '-'.join(chosen_obs_names)
    self.chosen_context_for_P = chosen_context_for_P
    self.seed = seed
    self.rng0 = np.random.default_rng(self.seed)
    self.rng1 = np.random.default_rng(self.seed)
    self.b_add_time_inhomogeneous = b_add_time_inhomogeneous
    self.b_using_uniform = b_using_uniform
    self.chosen_a = chosen_a
    self.δh = δh
    self.εh = εh
    self.δd = δd
    self.εd = εd
    self.µs = µs
    self.ρ1 = ρ1
    self.ρ2 = ρ2
    self.ρ3 = ρ3
    self.D_threshold = D_threshold
    self.init_c_true = 0
    self.init_probs = []
    for i in range(self.C):
      self.init_probs.append(1/self.C)
    self.init_c_infer = 0
    self.init_h = 0.1
    self.init_d = 0.1
    self.init_z = 0
    self.init_w = 1
    self.init_s = 0.1
    self.current_episode = -1
    
    self.reset()
    
    min_obs, max_obs = self.extract_min_max(self.chosen_obs_names, b_add_time_inhomogeneous)
    min_obs = np.array(min_obs)
    max_obs = np.array(max_obs)
    self.observation_space = spaces.Box(low=np.float32(min_obs), high=np.float32(max_obs), dtype=np.float32)
    self.action_space = spaces.Discrete(n_actions,)

    self.config = {'obs':self.chosen_obs_names_str,                   
                   'σ':sigma, 'δh':self.δh, 'εh':self.εh, 'δd':self.δd, 'εd':self.εd,
                   'µs':self.µs, 'ρ1':self.ρ1, 'ρ2':self.ρ2 , 'ρ3':self.ρ3, 'D_threshold':self.D_threshold,
                   'n_version':self.n_version, 'seed':self.seed}
    if self.n_version == 1:
      if self.b_using_uniform:
        self.config['a'] = self.chosen_a
      else:
        self.config['κh'] = self.κh
        self.config['κd'] = self.κd
      self.config['σs'] = self.σs

    if b_display:
      str_config = ', '.join([k + '='+ str(v) for k,v in self.config.items()])
      print('env config:', str_config)
    assert(len(µs)==n_context), 'error: length of µs must match the number of contexts.'

  def sample_context(self, chosen_rng, input_sigma, class_balance=.5):
    r"""This generates a context sample (with 2 contexts). The output is (c_true, p, c_infer), where:
    c_true is the true context, p is the context probabilities, and c_infer is the inferred context."""
    mu  = np.array([0,1])
    sigma_values = np.array([input_sigma, input_sigma])
    pc_true = np.array([class_balance, 1-class_balance])
    c_true = chosen_rng.choice(2, p=pc_true)
    x = mu[c_true] + sigma_values[c_true] * chosen_rng.standard_normal()
    if input_sigma > 0.1:
      p_num = norm.pdf((x-mu)/sigma_values) * pc_true
      p = p_num/(np.sum(p_num))
    else:
      p  = np.array([0,0])
      if c_true == 0:
        p[0] = 1
      else:
        p[1] = 1
    c_infer = np.argmax(p)
    return (c_true, p, c_infer)

  def create_obs_array(self, input_c_true, input_probs, input_c_infer, input_h, input_d, input_z, input_w, input_s):
    obs_array = [input_c_true]
    for i in range(self.C):
      obs_array.append(input_probs[i])
    obs_array.append(input_c_infer)
    obs_array.append(input_h)
    obs_array.append(input_d)
    obs_array.append(input_z)
    obs_array.append(input_w)
    obs_array.append(input_s)
    return np.array(obs_array)

  def unpack_obs_array(self,obs):
    input_c_true = int(obs[0])
    input_probs = obs[1:self.C+1]
    input_c_infer = int(obs[self.C+1])
    input_h = obs[self.C+2]
    input_d = obs[self.C+3]
    input_z = int(obs[self.C+4])
    input_w = int(obs[self.C+5])
    input_s = obs[self.C+6]
    return({'c_true':input_c_true, 'probs':input_probs, 'c_infer':input_c_infer, 'h':input_h, 'd':input_d, 'z':input_z, 'w':input_w, 's':input_s})

  def get_current_state(self):
    return self.current_state

  def get_C(self):
    return self.unpack_obs_array(self.current_state)['c_true']

  def get_H(self):
    return self.unpack_obs_array(self.current_state)['h']

  def get_P(self):
    return self.unpack_obs_array(self.current_state)['probs'][self.chosen_context_for_P]

  def get_L(self):
    return self.unpack_obs_array(self.current_state)['c_infer']

  def get_D(self):
    return self.unpack_obs_array(self.current_state)['d']

  def get_Z(self):
    return self.unpack_obs_array(self.current_state)['z']

  def get_W(self):
    return self.unpack_obs_array(self.current_state)['w']

  def get_S(self):
    return self.unpack_obs_array(self.current_state)['s']

  def get_T(self):
    indicator_value = 0 if ((self.current_t % 2) == 0) else 1
    return indicator_value

  def extract_obs(self, chosen_obs_names, b_add_time_inhomogeneous):
    r"""This extracts the obs in chosen_obs_names from the full state. If b_add_time_inhomogeneous is True,
    then the full state is augmented with the time inhomogeneous (one hot vector)."""
    obs_only = []
    for check_obs in chosen_obs_names:
      if check_obs == 'C':
        obs_only.append(self.get_C())
      elif check_obs == 'P':
        obs_only.append(self.get_P())
      elif check_obs == 'L':
        obs_only.append(self.get_L())
      elif check_obs == 'H':
        obs_only.append(self.get_H())
      elif check_obs == 'D':
        obs_only.append(self.get_D())
      elif check_obs == 'Z':
        obs_only.append(self.get_Z())
      elif check_obs == 'W':
        obs_only.append(self.get_W())
      elif check_obs == 'S':
        obs_only.append(self.get_S())
      elif check_obs == 'T':
        obs_only.append(self.get_T())
      else:
        str_message = 'error in check_obs. obs name {} does not exist.'.format(check_obs)
        print(str_message)
        assert(1==2), str_message
    if b_add_time_inhomogeneous:
      enc = OneHotEncoder(handle_unknown='ignore')
      X = np.arange(self.max_episode_length)
      X = X.reshape(-1, 1)
      enc.fit(X)
      one_hot  = list(enc.transform([[int(self.current_t)]]).toarray()[0])
      obs_only = obs_only + one_hot
    return obs_only

  def extract_min_max(self, chosen_obs_names, b_add_time_inhomogeneous):
    r"""This extracts the min and max values for the obs in chosen_obs_names. If b_add_time_inhomogeneous is True,
    then the results are augmented with the min and max values of the time inhomogeneous (one hot vector)."""
    min_values = []; max_values = []
    for check_obs in chosen_obs_names:
      if check_obs == 'C':
        min_values.append(0.)
        max_values.append(1.)
      elif check_obs == 'P':
        min_values.append(0.)
        max_values.append(1.)
      elif check_obs == 'L':
        min_values.append(0.)
        max_values.append(1.)
      elif check_obs == 'H':
        min_values.append(0.)
        max_values.append(1.)
      elif check_obs == 'D':
        min_values.append(0.)
        max_values.append(1.)
      elif check_obs == 'Z':
        min_values.append(0.)
        max_values.append(1.)
      elif check_obs == 'W':
        min_values.append(0.)
        max_values.append(1.)
      elif check_obs == 'S':
        min_values.append(0.)
        max_values.append(300.)
      elif check_obs == 'T':
        min_values.append(0.)
        max_values.append(1.)
      else:
        str_message = 'error in check_obs. obs name {} does not exist.'.format(check_obs)
        print(str_message)
        assert(1==2), str_message
    if b_add_time_inhomogeneous:
      min_values = min_values + list(np.zeros(self.max_episode_length))
      max_values = max_values + list(np.ones(self.max_episode_length))
    return min_values, max_values

  def reset(self):
    self.current_t = 0
    self.current_state = self.create_obs_array(self.init_c_true, self.init_probs, self.init_c_infer, self.init_h, self.init_d, self.init_z, self.init_w, self.init_s)
    self.obs = self.extract_obs(self.chosen_obs_names, self.b_add_time_inhomogeneous)
    if self.b_gymnasium:
      return self.obs, {}
    else:
      return self.obs

  def step(self, agent_action):
    a  = int(agent_action)
    ht = float(self.get_H())
    dt = float(self.get_D())
    st = float(self.get_S())
    obs_dict = self.unpack_obs_array(self.current_state)
    c_true = obs_dict["c_true"]
    if a == 0:
      h_next_mu = (1-self.δh) * ht
    else:
      h_next_mu = float(min(1, ht + self.εh))
    x  = 2 + c_true
    if a == 0:
      d_next_mu = dt
    elif (a == 1) or (a == x):
      d_next_mu = (1-self.δd) * dt
    else:
      d_next_mu = float(min(1, dt + self.εd))
    if   a == 0:
      s_next_mu = self.µs[c_true]
    elif a == 1:
      s_next_mu = self.µs[c_true] + (1 - h_next_mu) * self.ρ1
    elif a == x:
      s_next_mu = self.µs[c_true] + (1 - h_next_mu) * self.ρ2
    else:
      s_next_mu = self.µs[c_true] + (1 - h_next_mu) * self.ρ3

    c_true, probs_next_mu, c_infer_next_mu = self.sample_context(self.rng0, self.sigma)

    if self.n_version == 0:
      # deterministic version
      probs_next = probs_next_mu
      c_infer_next = c_infer_next_mu
      h_next = h_next_mu
      d_next = d_next_mu
      s_next = s_next_mu
    elif self.n_version == 1:
      # stochastic version
      probs_next = probs_next_mu
      c_infer_next = c_infer_next_mu
      #------------------------------------------------------
      if self.b_using_uniform:
        # using Uniform distribution
        lower_h = (1-self.chosen_a/2)*h_next_mu
        upper_h = (1+self.chosen_a/2)*h_next_mu
        h_next = float(self.rng1.uniform(lower_h,upper_h,size=1))

        lower_d = (1-self.chosen_a/2)*d_next_mu
        upper_d = (1+self.chosen_a/2)*d_next_mu
        d_next = float(self.rng1.uniform(lower_d,upper_d,size=1))
      else:
        # using Beta distribution
        a_h = float(max(1e-5, self.κh * h_next_mu))
        b_h = float(max(1e-5, self.κh * (1-h_next_mu)))
        h_next = float(self.rng1.beta(a_h, b_h, size=1))

        a_d = float(max(1e-5, self.κd * d_next_mu))
        b_d = float(max(1e-5, self.κd * (1-d_next_mu)))
        d_next = float(self.rng1.beta(a_d, b_d, size=1))
      shape_k = (float(max(1e-5, s_next_mu)) / self.σs)**2
      scale_theta = (self.σs)**2 / float(max(1e-5, s_next_mu))
      s_next = float(self.rng1.gamma(shape=shape_k, scale=scale_theta, size=1))
      #------------------------------------------------------
      c_infer_next = min(1,c_infer_next); c_infer_next = max(0,c_infer_next)
      for i in range(self.C):
        probs_next[i] = float(min(1,probs_next[i])); probs_next[i] = float(max(0,probs_next[i]))
    else:
      error_message = 'Attention! env n_version = {} does not exit!'.format(self.n_version)
      print(error_message)
      assert(1==2), error_message
    # clip for deterministic and stochastic dynamics
    h_next = float(min(1,h_next)); h_next = float(max(0,h_next))
    d_next = float(min(1,d_next)); d_next = float(max(0,d_next))
    s_next = float(max(0,s_next))
    z_next = c_true   # TO DO
    w_next = c_true   # TO DO
    step_reward = s_next
    current_state = self.create_obs_array(c_true, probs_next, c_infer_next, h_next, d_next, z_next, w_next, s_next)
    self.current_state = current_state
    self.obs = self.extract_obs(self.chosen_obs_names, self.b_add_time_inhomogeneous)
    last_t = self.current_t

    # prepare for next step
    self.current_t += 1

    condition1 = (last_t >= self.max_episode_length)
    condition2 = (d_next >= self.D_threshold)
    if condition1 or condition2:
      info = {}
      if condition1:
        info['done'] = 'found t={} (> max episode length)'.format(last_t, self.max_episode_length)
      if condition2:
        info['done'] = 'found d={:.1f} (> disengage threshold={}, at t={})'.format(d_next, self.D_threshold, last_t)
      done = True

      # if done, set reward to 0 (set step count to 0)
      old_reward = step_reward
      s_next = 0
      step_reward = s_next
      current_state = self.create_obs_array(c_true, probs_next, c_infer_next, h_next, d_next, z_next, w_next, s_next)
      self.current_state = current_state
      self.obs = self.extract_obs(self.chosen_obs_names, self.b_add_time_inhomogeneous)
      if self.b_gymnasium:
        return self.obs, step_reward, done, done, info
      else:
        return self.obs, step_reward, done, info
    else:
      done = False
      if self.b_gymnasium:
        return self.obs, step_reward, done, done, {}
      else:
        return self.obs, step_reward, done, {}

  def get_inferred_error(self, chosen_sigma, b_default=True, sample_seed=0, N_data=1000):
    r"""This computes the corresponding inferred error, given the uncertainty sigma.
    If b_default is True then this uses some default values. If b_default is False then this computes the inferred error,
    using N_data samples and sample_seed."""
    if b_default:
      map_sigma_error = {'0':0., '0.4':0.096, '0.6':.175, '0.8':.273, '1':.311, '2':.414}
      inferred_error = map_sigma_error[str(chosen_sigma)]
    else:
      c_true  = np.zeros((N_data,),dtype=int)
      c_infer = np.zeros((N_data,),dtype=int)
      chosen_rng = np.random.default_rng(sample_seed)
      for i in range(N_data):
        c_true[i], _ ,c_infer[i] = self.sample_context(chosen_rng, chosen_sigma)
      inferred_error = np.mean(c_true!=c_infer)
    return inferred_error

  def get_current_state_length(self):
    return len(self.current_state)

  def get_obs_length(self):
    return len(self.obs)

  def render(self):
    pass

