from src.environment.usv_environment import USVEnvironment
from src.training.train_residual_sac import train_residual_sac


env = USVEnvironment()
agent, history = train_residual_sac(
    env,
    episodes=20,
    warmup_steps=500,
    batch_size=64,
)

print("Training complete.")
print("Episodes:", len(history))
