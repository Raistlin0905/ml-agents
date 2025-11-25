import json
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
)
from uuid import UUID

received_json_string = ""


class EnvSideChannel(SideChannel):

    def __init__(self):
        super().__init__(UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage):
        global received_json_string
        received_json_string = msg.read_string()

    def get_attributes(self):
        if not received_json_string:
            print("No data received on side channel. Returning empty dict.")
            return {}
        try:
            return json.loads(received_json_string)
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e, "Raw data:", received_json_string)
            return {}
