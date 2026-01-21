from mlagents_envs.environment import UnityEnvironment
from env_side_channel import EnvSideChannel


class EnvCollector:

    def __init__(self, run_id, env_name):
        self.run_id = run_id
        self.env_name = env_name

    def collect_data(self) -> dict:
        channel = EnvSideChannel()
        unity_env = UnityEnvironment(file_name=None, side_channels=[channel])

        unity_env.reset()

        behaviors = sorted(list(unity_env.behavior_specs.keys()))
        first_behavior = behaviors[0]
        spec = unity_env.behavior_specs[first_behavior]

        attributes = {
            "behavior_name": first_behavior,
            "actions_continuous_actions": spec.action_spec.continuous_size,
            "actions_discrete_size": spec.action_spec.discrete_size,
            "actions_discrete_branches": ";".join(
                map(str, spec.action_spec.discrete_branches)
            ),
            "model": "",
            "inference_device": "",
            "deterministic_inference": "",
            "behavior_type": "",
            "team_id": "",
            "use_child_actuators": "",
            "use_child_sensors": "",
            "observational_attribute_handling": "",
            "max_step": "",
            "decision_period": "",
            "decision_step": "",
            "take_actions_between_decisions": "",
        }

        unity_env.close()

        unity_attributes = channel.get_attributes()

        attributes.update(unity_attributes)

        return attributes
