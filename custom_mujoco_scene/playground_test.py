import jax
from mujoco_playground import registry

# Load the environment
env = registry.load('CartpoleBalance')

# JIT-compile the simulation functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# Initialize the environment and perform a rollout
state = jit_reset(jax.random.PRNGKey(0))
rollout = [state]

# Define and apply your actions here
for _ in range(env.cfg.episode_length):
    action = ... # Your policy for action selection
    state = jit_step(state, action)
    rollout.append(state)


"""

#### Vision-based policies
For training agents using pixel observations, MuJoCo Playground integrates with the Madrona batch renderer.
*   The Madrona renderer is GPU-based and supports features like shadows, textures, and lighting.
*   The integration allows for end-to-end training of vision-based policies on a single GPU.
*   For effective sim-to-real transfer, the visual properties of environments can be randomized, including object color, lighting, and camera pose.

#### Sim-to-real transfer
MuJoCo Playground is designed for practical robotics research, with a focus on transferring policies trained in simulation to real-world robots.
*   It supports a variety of robot platforms, such as quadrupeds, humanoids, and robotic arms, many of which have demonstrated successful zero-shot transfer.
*   Techniques like domain randomization are used to ensure the trained policies are robust to real-world uncertainties.

### Resources

*   **Official Website**: The main resource for news and project information is playground.mujoco.org.
*   **GitHub Repository**: Find the source code, open issues, and developer documentation at github.com/google-deepmind/mujoco_playground.
*   **Official Colab Notebooks**: The project offers Colab notebooks for getting started with locomotion and manipulation tasks, providing a zero-setup starting point.

"""