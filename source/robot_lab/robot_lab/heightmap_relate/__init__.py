from .heightmap_actor_critic import HeightMapActorCritic


def register_heightmap_actor_critic():
    """Expose HeightMapActorCritic to RSL-RL's string-based policy resolver."""

    import rsl_rl.modules as rsl_rl_modules
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    rsl_rl_modules.HeightMapActorCritic = HeightMapActorCritic
    on_policy_runner.HeightMapActorCritic = HeightMapActorCritic


__all__ = ["HeightMapActorCritic", "register_heightmap_actor_critic"]
