import os
import yaml


class ConfigCollector:
    """Collects training configuration from YAML files."""

    def __init__(self, config_base_dir: str = "config"):
        self.config_base_dir = config_base_dir

    def collect_data(self, trainer_type: str, env_name: str) -> dict:

        yaml_path = os.path.join(self.config_base_dir, trainer_type, f"{env_name}.yaml")
        return self.collect_from_path(yaml_path)

    def collect_from_path(self, yaml_path: str) -> dict:
        """Collect config data from a specific YAML file path."""
        if not os.path.exists(yaml_path):
            return self._get_empty_config()

        try:
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)
        except (yaml.YAMLError, IOError):
            return self._get_empty_config()

        return self._extract_fields(config)

    def _get_empty_config(self) -> dict:
        """Return config dict with all fields set to None."""
        return {
            "trainer_type": None,
            "batch_size": None,
            "buffer_size": None,
            "learning_rate": None,
            "beta": None,
            "epsilon": None,
            "lambd": None,
            "num_epoch": None,
            "shared_critic": None,
            "learning_rate_schedule": None,
            "beta_schedule": None,
            "epsilon_schedule": None,
            "checkpoint_interval": None,
            "keep_checkpoints": None,
            "even_checkpoints": None,
            "normalize": None,
            "hidden_units": None,
            "num_layers": None,
            "vis_encode_type": None,
            "memory": None,
            "goal_conditioning_type": None,
            "deterministic": None,
            "extrinsic_gamma": None,
            "extrinsic_strength": None,
            "init_path": None,
            "threaded": None,
            "max_steps": None,
            "time_horizon": None,
            "summary_freq": None,
            "self_play": None,
            "behavioral_cloning": None,
        }

    def _extract_fields(self, config: dict) -> dict:
        """Extract relevant fields from parsed YAML config."""
        result = self._get_empty_config()

        behaviors = config.get("behaviors", {})
        if not behaviors:
            return result

        behavior_name = list(behaviors.keys())[0]
        behavior = behaviors[behavior_name]

        default_settings = config.get("default_settings", {})
        behavior = self._deep_merge(default_settings, behavior)

        hyperparams = behavior.get("hyperparameters", {})
        network = behavior.get("network_settings", {})
        reward_signals = behavior.get("reward_signals", {})
        extrinsic = reward_signals.get("extrinsic", {})

        result["trainer_type"] = behavior.get("trainer_type")
        result["batch_size"] = hyperparams.get("batch_size")
        result["buffer_size"] = hyperparams.get("buffer_size")
        result["learning_rate"] = hyperparams.get("learning_rate")
        result["beta"] = hyperparams.get("beta")
        result["epsilon"] = hyperparams.get("epsilon")
        result["lambd"] = hyperparams.get("lambd")
        result["num_epoch"] = hyperparams.get("num_epoch")
        result["shared_critic"] = hyperparams.get("shared_critic")

        result["learning_rate_schedule"] = hyperparams.get("learning_rate_schedule")
        result["beta_schedule"] = hyperparams.get("beta_schedule")
        result["epsilon_schedule"] = hyperparams.get("epsilon_schedule")

        result["checkpoint_interval"] = behavior.get("checkpoint_interval")
        result["keep_checkpoints"] = behavior.get("keep_checkpoints")
        result["even_checkpoints"] = behavior.get("even_checkpoints")

        result["normalize"] = network.get("normalize")
        result["hidden_units"] = network.get("hidden_units")
        result["num_layers"] = network.get("num_layers")
        result["vis_encode_type"] = network.get("vis_encode_type")

        memory = network.get("memory")
        if memory:
            result["memory"] = str(memory)

        result["goal_conditioning_type"] = network.get("conditioning_type")
        result["deterministic"] = behavior.get("deterministic")

        result["extrinsic_gamma"] = extrinsic.get("gamma")
        result["extrinsic_strength"] = extrinsic.get("strength")

        result["init_path"] = behavior.get("init_path")
        result["threaded"] = behavior.get("threaded")

        result["max_steps"] = behavior.get("max_steps")
        result["time_horizon"] = behavior.get("time_horizon")
        result["summary_freq"] = behavior.get("summary_freq")

        self_play = behavior.get("self_play")
        if self_play:
            result["self_play"] = str(self_play)

        behavioral_cloning = behavior.get("behavioral_cloning")
        if behavioral_cloning:
            result["behavioral_cloning"] = str(behavioral_cloning)

        return result

    def _deep_merge(self, base: dict, override: dict) -> dict:
        result = dict(base)
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = self._deep_merge(result[key], val)
            else:
                result[key] = val
        return result
