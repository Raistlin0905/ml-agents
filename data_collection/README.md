# Data Collection Script
Script to collect hardware data (CPU, disk, GPU, RAM specs) and environmental data (agent behavior parameters including steps, inference device, and use of child sensors) and training configuration data from config/yaml files (including trainer type, learning rate, and summary frequency).

# Prerequisites
Packages list:
```
# Windows
ml-agents/reqs/windows/requirements.txt

# Macos
ml-agents/reqs/macos/requirements.txt
```

# How to Run
Usage: 
```
python {path/to/data_collection/main.py} {docker_used: true|false} {trainer: imitation|ppo|sac|poca} {env_name: short game name without file extension (Crawler/Hallway/Pyramids/3DBall etc.)}
```

To run the data_collection script, run the command above, and once the terminal hangs, run the same game in unity. Afterwards, the collected data will be written to the `human_readable.csv` file.

# Sample Run
1. Current location is data_collection
2. Target scene is opened in Unity Hierarchy (in this case, Crawler)

```
python main.py false ppo Crawler
```

3. Program waits for Unity connection through the side channel
4. Hit the play button to run Unity scene; collector collects environment data
5. Script writes data to csv and terminates successfully
6. New data can be found at `human_readable.csv` in the data_collection directory

# Classes
## main.py
CLI entry which orchestrates collectors and writes rows using the CSVWriter.

## csv_writer.py
Opens and writes to csv file; writes header if new file.

## base_collector.py
Data-collection abstract class which defines the methods necessary to collect data from allowed systems.

## cpu_collector.py
Inherits from base collector to collect cpu data from host system.

## ram_collector.py
Inherits from base collector to collect ram data from host system.

## disk_collector.py
Collects disk data from host system.

## disk_adaptor.py
Inherits from base collector to map disk_collector results to base_collector-defined format.

## gpu_collector.py
Collects gpu data from host system.

## gpu_adaptor.py
Inherits from base collector to map gpu_collector results to base_collector-defined format and flatten nested dictionaries.

## env_collector.py
Collects agent behavior data through python<->unity side channel using env_side_channel.py.

## env_side_channel.py
Defines a common ID recognizable by unity's side channel and returns message received from unity.

## yaml_config_collector.py
Parses yaml configuration file for info and returns flatten dictionary.