import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

class van(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()


    z: Any 
    a: Any 

    def expectation(self,probs):
        if self.config["num_atoms"] == 1: return probs
        else:  return jnp.sum(probs * self.z, axis=-1)


    def expectation_01(self,probs):
        if self.config["num_atoms"] == 1: return probs
        else:  return jnp.sum(probs * self.a, axis=-1)
        return expected




    def cvar(self, probs, alpha=0.1):
        """
        returns:
            expected_value: (batch,)
            variance: (batch,)
            cvar: (batch,)
        """


        if self.config["num_atoms"] == 1:
            expected = probs
            variance = jnp.zeros_like(probs)
            cvar = probs
            return expected, variance, cvar

        # support
        atoms = self.z  # (num_atoms,)


        expected = jnp.sum(probs * atoms, axis=-1)


        expected_sq = jnp.sum(probs * (atoms ** 2), axis=-1)
        variance = expected_sq - expected ** 2


        cdf = jnp.cumsum(probs, axis=-1)


        mask = cdf <= alpha


        tail_probs = probs * mask
        tail_mass = jnp.sum(tail_probs, axis=-1, keepdims=True)


        cvar = jnp.sum(tail_probs * atoms, axis=-1) / jnp.squeeze(
            jnp.maximum(tail_mass, 1e-8)
        )

        return expected, variance, cvar


    def expectation_01_risk(self, probs):
        if self.config["num_atoms"] == 1:
            return probs

        z = self.a
        cdf = jnp.cumsum(probs, axis=-1)

        weights = (1.0 - cdf) ** self.config["delta"]

        weighted_probs = probs * weights
        weighted_probs /= jnp.sum(weighted_probs, axis=-1, keepdims=True) + 1e-8

        return jnp.sum(weighted_probs * z, axis=-1)





    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action
        
        # TD loss
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        next_qs = self.network.select(f'target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)
        
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)
        
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def critic_loss_dist(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action
        
        # TD loss
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)



        next_qs = self.network.select(f'target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)


        if self.config['q_agg'] == 'min':
            next_expectation_qs = self.expectation(next_qs)

            min_idx = jnp.argmin(next_expectation_qs, axis=0)  # (B,)
            min_idx = min_idx[None, :, None]

            next_q_dist = jnp.take_along_axis(
                next_qs,
                min_idx,
                axis=0
            ).squeeze(axis=0)  # (B, num_atoms)
        else:
            next_q_dist = jnp.mean(next_qs, axis=0)
            

        target = (self.config['discount']**self.config['horizon_length']) * batch['masks'][..., -1]


        target_dist = self.categorical_projection(next_q_dist,batch['rewards'][..., -1],target)

        q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)


        eps = 1e-8

        critic_loss = -(jnp.sum(target_dist[None, :, :] * jnp.log(q + eps),axis=-1 )* batch['valid'][..., -1] ).mean()  # (B,)
 
        # jax.debug.print("reward = {}", batch['rewards'][..., -1].max())

        exp, var, cvar = self.cvar(next_qs)
        # cvar = self.expectation_01_risk(next_qs)
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': exp.mean(),
            'q_max': exp.max(),
            'q_min': exp.min(),
            'var': var.mean(),
            'cvar': cvar.mean(),
        }



    def categorical_projection(self, next_dist, rewards, discounts):
        """
        next_dist: (B, num_atoms) probabilities
        rewards: (B,)
        discounts: (B,)
        returns: m (B, num_atoms)
        """
        B = rewards.shape[0]
        num_atoms = self.config['num_atoms']
        v_min = self.config['v_min']
        v_max = self.config['v_max']
        z = jnp.linspace(v_min, v_max, num_atoms)
        delta_z = (v_max - v_min) / (num_atoms - 1)

        # Tz: (B, num_atoms)
        Tz = rewards[:, None] + discounts[:, None] * z[None, :]
        Tz = jnp.clip(Tz, v_min, v_max)

        b = (Tz - v_min) / delta_z  # (B, num_atoms)
        l = jnp.floor(b).astype(jnp.int32)
        u = jnp.ceil(b).astype(jnp.int32)

        # clip
        l = jnp.clip(l, 0, num_atoms - 1)
        u = jnp.clip(u, 0, num_atoms - 1)

        m = jnp.zeros((B, num_atoms))

        batch_idx = jnp.arange(B)[:, None]
        batch_idx = jnp.broadcast_to(batch_idx, l.shape)

        # weight to lower and upper bins
        upper_weight = (b - l)  # (B, num_atoms)
        lower_weight = (u - b)  # (B, num_atoms)

        # When l == u, lower_weight == upper_weight == 0, so we want full mass to that bin.
        # So handle integer case explicitly:
        same = (l == u)
        # add lower
        m = m.at[(batch_idx, l)].add(next_dist * (lower_weight + same.astype(jnp.float32)))
        # add upper
        m = m.at[(batch_idx, u)].add(next_dist * (upper_weight))

        # ensure numerical normalization (optional)
        m = m / (m.sum(axis=-1, keepdims=True) + 1e-12)

        return m

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)



        if self.config["actor_type"] == "distill-ddpg":
            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
            
            # Q loss.
            actor_actions = jnp.clip(actor_actions, -1, 1)

            qs = self.network.select(f'critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)
            q_loss = -q.mean()
        else:
            if self.config["q_guidance"]:
                actions = x_t + pred / self.config['flow_steps']
                actions = jnp.clip(actions, -1, 1)
                qs = self.network.select('critic')(batch['observations'], actions=actions)
                if self.config["risk"]: qs = self.expectation_01_risk(qs)
                else : qs = self.expectation_01(qs)
                q = jnp.mean(qs, axis=0)



                if self.config["num_atoms"] == 1:
                    q_abs_mean = jnp.mean(jnp.abs(q))    # scalar
                    q_loss = - jnp.mean(q) / (q_abs_mean)

                else : q_loss =  - q.mean()

            else :

                q_loss = jnp.zeros(())
            distill_loss = jnp.zeros(())
        # only bc on the valid chunk indices
        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2 , 
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
                ) * batch["valid"][..., None] 
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        # Total loss.
        # actor_loss =   self.config['lmbda'] * bc_flow_loss + self.config['alpha'] * distill_loss + q_loss
        actor_loss =   self.config['lmbda']* bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            # 'weight': weight.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        if self.config["num_atoms"] == 1:
            critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        else :
            critic_loss, critic_info = self.critic_loss_dist(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @jax.jit

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        
        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1),
                ),
            )
            actions = self.network.select(f'actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
            action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config["actor_num_samples"], action_dim
                ),
            )
            observations = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
            actions = self.compute_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            q = self.network.select("critic")(observations, actions)
            if self.config["risk"]: q = self.expectation_01_risk(q)
            else : q = self.expectation_01(q)
            if self.config["q_agg"] == "mean":
                q = q.mean(axis=0)
            else:
                q = q.min(axis=0)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config["actor_num_samples"], action_dim))[jnp.arange(bsize), indices, :].reshape(
                bshape + (action_dim,))

        return actions




    @jax.jit
    def sample_target_actions(
        self,
        observations,
        rng=None,
    ):
        
        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1),
                ),
            )
            actions = self.network.select(f'actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
            action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config["actor_num_samples"], action_dim
                ),
            )
            observations = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
            actions = self.compute_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            q = self.network.select("target_critic")(observations, actions)
            if self.config["risk"]: q_exp = self.expectation_01_risk(q)
            else : q_exp = self.expectation_01(q)
            if self.config["q_agg"] == "mean":
                q_exp = q_exp.mean(axis=0)
            else:
                q_exp = q_exp.min(axis=0)
            indices = jnp.argmax(q_exp, axis=-1)
            indices = indices[None, :, None, None]
            best_q = jnp.take_along_axis(
                q,
                indices,
                axis=2
            ).squeeze(2)


        return best_q



    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def bc_flow_loss_(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, bc_flow_rng = jax.random.split(rng, 2)

        bc_flow_loss, bc_flow_info = self.bc_flow_loss(batch, grad_params, bc_flow_rng)
        for k, v in bc_flow_info.items():
            info[f'bc_flow/{k}'] = v


        loss = bc_flow_loss
        return loss, info
    def bc_flow_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)

        # only bc on the valid chunk indices
        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2, 
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
                ) * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        return bc_flow_loss, {
            'bc_flow_loss': bc_flow_loss,
        }


    def compute_bc_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions
    @staticmethod
    def _bc_flow_update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.bc_flow_loss_(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        return agent.replace(network=new_network, rng=new_rng), info
    @jax.jit
    def bc_flow_update(self, batch ):
        return self._bc_flow_update(self, batch )


    def bctoactor(self):
        src = self.network.params['modules_actor_bc_flow']
        dst = self.network.params['modules_actor_flow']

        new_actor_flow = jax.tree_util.tree_map(
            lambda p, _: p,
            src,
            dst,
        )

        self.network.params['modules_actor_flow'] = new_actor_flow


    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        z = jnp.linspace(config['v_min'], config['v_max'], config['num_atoms'])
        a = jnp.linspace(config['v_min'], config['v_max'], config['num_atoms'])
        # a = jnp.linspace(-config['lmbda'], 0, config['num_atoms'])

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
            num_atoms = config['num_atoms']
        )

        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )

        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        
        network_info = dict(
            actor_flow=(actor_flow_def, (ex_observations, full_actions, ex_times)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_flow') is not None:
            # Add actor_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_flow_encoder'] = (encoders.get('actor_flow'), (ex_observations,))
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        params[f'modules_target_critic'] = params[f'modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config),z=z,a=a,)


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='van',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.995,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=100.0,  # BC coefficient (need to be tuned for each environment).
            num_qs=2, # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=False,  # False means n-step return
            actor_type="best-of-n",
            actor_num_samples=4,  # for actor_tyfpe="best-of-n" only
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            num_atoms=101,
            v_min = -200,
            v_max = 0,
            q_guidance = True,
            lmbda = 3,
            bcflowstep = 0,
            risk = True,
            delta = 2,
        )
    )
    return config
