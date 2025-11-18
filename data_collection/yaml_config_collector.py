from typing import Dict
import yaml


class YAMLConfigCollector:
    def load_yaml(self, path: str):
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def get_features(self):
        return {
            "behavior_count": "",
            "behaviors0_trainer_type": "",
            "behaviors0_summary_freq": "",
            "behaviors0_time_horizon": "",
            "behaviors0_max_steps": "",
            "behaviors0_keep_checkpoints": "",
            "behaviors0_even_checkpoints": "",
            "behaviors0_checkpoint_interval": "",
            "behaviors0_threaded": "",
            "behaviors0_hyperparameters_learning_rate": "",
            "behaviors0_hyperparameters_batch_size": "",
            "behaviors0_hyperparameters_buffer_size": "",
            "behaviors0_hyperparameters_learning_rate_schedule": "",
            "behaviors0_hyperparameters_beta": "",
            "behaviors0_hyperparameters_epsilon": "",
            "behaviors0_hyperparameters_beta_schedule": "",
            "behaviors0_hyperparameters_epsilon_schedule": "",
            "behaviors0_hyperparameters_lambd": "",
            "behaviors0_hyperparameters_num_epoch": "",
            "behaviors0_hyperparameters_shared_critic": "",
            "behaviors0_hyperparameters_buffer_init_steps": "",
            "behaviors0_hyperparameters_init_entcoef": "",
            "behaviors0_hyperparameters_save_replay_buffer": "",
            "behaviors0_hyperparameters_tau": "",
            "behaviors0_hyperparameters_steps_per_update": "",
            "behaviors0_hyperparameters_num_update": "",
            "behaviors0_network_settings_hidden_units": "",
            "behaviors0_network_settings_num_layers": "",
            "behaviors0_network_settings_normalize": "",
            "behaviors0_network_settings_vis_encode_type": "",
            "behaviors0_network_settings_conditioning_type": "",
            "behaviors0_network_settings_memory_size": "",
            "behaviors0_network_settings_sequence_length": "",
            "behaviors0_reward_signals_extrinsic_strength": "",
            "behaviors0_reward_signals_extrinsic_gamma": "",
            "behaviors0_reward_signals_curiosity_strength": "",
            "behaviors0_reward_signals_curiosity_gamma": "",
            "behaviors0_reward_signals_curiosity_network_settings_hidden_units": "",
            "behaviors0_reward_signals_curiosity_network_settings_num_layers": "",
            "behaviors0_reward_signals_curiosity_network_settings_normalize": "",
            "behaviors0_reward_signals_curiosity_network_settings_vis_encode_type": "",
            "behaviors0_reward_signals_curiosity_network_settings_conditioning_type": "",
            "behaviors0_reward_signals_curiosity_network_settings_memory_memory_size": "",
            "behaviors0_reward_signals_curiosity_network_settings_memory_sequence_length": "",
            "behaviors0_reward_signals_curiosity_learning_rate": "",
            "behaviors0_reward_signals_gail_strength": "",
            "behaviors0_reward_signals_gail_gamma": "",
            "behaviors0_reward_signals_gail_network_settings_hidden_units": "",
            "behaviors0_reward_signals_gail_network_settings_num_layers": "",
            "behaviors0_reward_signals_gail_network_settings_normalize": "",
            "behaviors0_reward_signals_gail_network_settings_vis_encode_type": "",
            "behaviors0_reward_signals_gail_network_settings_conditioning_type": "",
            "behaviors0_reward_signals_gail_network_settings_memory_memory_size": "",
            "behaviors0_reward_signals_gail_network_settings_memory_sequence_length": "",
            "behaviors0_reward_signals_gail_learning_rate": "",
            "behaviors0_reward_signals_gail_use_actions": "",
            "behaviors0_reward_signals_gail_use_vail": "",
            "behaviors0_reward_signals_rnd_strength": "",
            "behaviors0_reward_signals_rnd_gamma": "",
            "behaviors0_reward_signals_rnd_network_settings_hidden_units": "",
            "behaviors0_reward_signals_rnd_network_settings_num_layers": "",
            "behaviors0_reward_signals_rnd_network_settings_normalize": "",
            "behaviors0_reward_signals_rnd_network_settings_vis_encode_type": "",
            "behaviors0_reward_signals_rnd_network_settings_conditioning_type": "",
            "behaviors0_reward_signals_rnd_network_settings_memory_memory_size": "",
            "behaviors0_reward_signals_rnd_network_settings_memory_sequence_length": "",
            "behaviors0_reward_signals_rnd_learning_rate": "",
            "behaviors0_behavioral_cloning_strength": "",
            "behaviors0_behavioral_cloning_steps": "",
            "behaviors0_behavioral_cloning_batch_size": "",
            "behaviors0_behavioral_cloning_num_epoch": "",
            "behaviors0_behavioral_cloning_samples_per_update": "",
            "behaviors0_self_play_save_steps": "",
            "behaviors0_self_play_team_change": "",
            "behaviors0_self_play_swap_steps": "",
            "behaviors0_self_play_play_against_latest_model_ratio": "",
            "behaviors0_self_play_window": "",
            "behaviors1_trainer_type": "",
            "behaviors1_summary_freq": "",
            "behaviors1_time_horizon": "",
            "behaviors1_max_steps": "",
            "behaviors1_keep_checkpoints": "",
            "behaviors1_even_checkpoints": "",
            "behaviors1_checkpoint_interval": "",
            "behaviors1_threaded": "",
            "behaviors1_hyperparameters_learning_rate": "",
            "behaviors1_hyperparameters_batch_size": "",
            "behaviors1_hyperparameters_buffer_size": "",
            "behaviors1_hyperparameters_learning_rate_schedule": "",
            "behaviors1_hyperparameters_beta": "",
            "behaviors1_hyperparameters_epsilon": "",
            "behaviors1_hyperparameters_beta_schedule": "",
            "behaviors1_hyperparameters_epsilon_schedule": "",
            "behaviors1_hyperparameters_lambd": "",
            "behaviors1_hyperparameters_num_epoch": "",
            "behaviors1_hyperparameters_shared_critic": "",
            "behaviors1_hyperparameters_buffer_init_steps": "",
            "behaviors1_hyperparameters_init_entcoef": "",
            "behaviors1_hyperparameters_save_replay_buffer": "",
            "behaviors1_hyperparameters_tau": "",
            "behaviors1_hyperparameters_steps_per_update": "",
            "behaviors1_hyperparameters_num_update": "",
            "behaviors1_network_settings_hidden_units": "",
            "behaviors1_network_settings_num_layers": "",
            "behaviors1_network_settings_normalize": "",
            "behaviors1_network_settings_vis_encode_type": "",
            "behaviors1_network_settings_conditioning_type": "",
            "behaviors1_network_settings_memory_size": "",
            "behaviors1_network_settings_sequence_length": "",
            "behaviors1_reward_signals_extrinsic_strength": "",
            "behaviors1_reward_signals_extrinsic_gamma": "",
            "behaviors1_reward_signals_curiosity_strength": "",
            "behaviors1_reward_signals_curiosity_gamma": "",
            "behaviors1_reward_signals_curiosity_network_settings_hidden_units": "",
            "behaviors1_reward_signals_curiosity_network_settings_num_layers": "",
            "behaviors1_reward_signals_curiosity_network_settings_normalize": "",
            "behaviors1_reward_signals_curiosity_network_settings_vis_encode_type": "",
            "behaviors1_reward_signals_curiosity_network_settings_conditioning_type": "",
            "behaviors1_reward_signals_curiosity_network_settings_memory_memory_size": "",
            "behaviors1_reward_signals_curiosity_network_settings_memory_sequence_length": "",
            "behaviors1_reward_signals_curiosity_learning_rate": "",
            "behaviors1_reward_signals_gail_strength": "",
            "behaviors1_reward_signals_gail_gamma": "",
            "behaviors1_reward_signals_gail_network_settings_hidden_units": "",
            "behaviors1_reward_signals_gail_network_settings_num_layers": "",
            "behaviors1_reward_signals_gail_network_settings_normalize": "",
            "behaviors1_reward_signals_gail_network_settings_vis_encode_type": "",
            "behaviors1_reward_signals_gail_network_settings_conditioning_type": "",
            "behaviors1_reward_signals_gail_network_settings_memory_memory_size": "",
            "behaviors1_reward_signals_gail_network_settings_memory_sequence_length": "",
            "behaviors1_reward_signals_gail_learning_rate": "",
            "behaviors1_reward_signals_gail_use_actions": "",
            "behaviors1_reward_signals_gail_use_vail": "",
            "behaviors1_reward_signals_rnd_strength": "",
            "behaviors1_reward_signals_rnd_gamma": "",
            "behaviors1_reward_signals_rnd_network_settings_hidden_units": "",
            "behaviors1_reward_signals_rnd_network_settings_num_layers": "",
            "behaviors1_reward_signals_rnd_network_settings_normalize": "",
            "behaviors1_reward_signals_rnd_network_settings_vis_encode_type": "",
            "behaviors1_reward_signals_rnd_network_settings_conditioning_type": "",
            "behaviors1_reward_signals_rnd_network_settings_memory_memory_size": "",
            "behaviors1_reward_signals_rnd_network_settings_memory_sequence_length": "",
            "behaviors1_reward_signals_rnd_learning_rate": "",
            "behaviors1_behavioral_cloning_strength": "",
            "behaviors1_behavioral_cloning_steps": "",
            "behaviors1_behavioral_cloning_batch_size": "",
            "behaviors1_behavioral_cloning_num_epoch": "",
            "behaviors1_behavioral_cloning_samples_per_update": "",
            "behaviors1_self_play_save_steps": "",
            "behaviors1_self_play_team_change": "",
            "behaviors1_self_play_swap_steps": "",
            "behaviors1_self_play_play_against_latest_model_ratio": "",
            "behaviors1_self_play_window": "",
        }

    def flatten_dict(self, old_dict: Dict[any, any]) -> dict:
        flat_dict = {}
        stack = [(old_dict, "")]

        while stack:
            current_dict, prefix = stack.pop()

            for key, val in current_dict.items():
                new_key = f"{prefix}_{key}" if prefix else key

                if isinstance(val, list):
                    for index, item in enumerate(val):
                        if isinstance(item, dict):
                            stack.append((item, f"{new_key}{index}"))
                elif isinstance(val, dict):
                    stack.append((val, new_key))
                else:
                    flat_dict[new_key] = val

        return flat_dict

    def collect_data(self, yaml_path: str, env_name: str):
        path = yaml_path

        file = self.load_yaml(path)
        features = self.get_features()

        behaviors = file.get("behaviors")
        features["behavior_count"] = len(behaviors)

        for index, (_, value) in enumerate(behaviors.items()):
            if index > 1:
                break
            prefix = f"behaviors{index}_"
            flat = self.flatten_dict(value)

            for key, val in flat.items():
                new_key = prefix + key
                if new_key in features:
                    features[new_key] = val

        return features
