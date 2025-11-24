from typing import TypedDict, Optional


class TrainingRunData(TypedDict):
    # Identifiers
    run_id: int

    # System/Hardware Features
    host_cpu_features: str
    host_ram_features: str
    host_gpu_features: str
    host_os_name: str
    host_disk_features: str
    is_docker_used: bool

    # Environment
    environment_features: str
    reward_density: float

    # Core Training Hyperparameters
    trainer_type: str
    batch_size: int
    buffer_size: int
    learning_rate: float
    beta: float
    epsilon: float
    lamdb: float
    num_epoch: int
    shared_critic: bool

    # Schedules
    learning_rate_schedule: str
    beta_schedule: str
    epsilon_schedule: str

    # Checkpointing
    checkpoint_interval: int

    # Network Architecture
    normalize: bool
    hidden_units: int
    num_layers: int
    vis_encode_type: str

    # Advanced Features
    memory: Optional[str]
    goal_conditioning_type: str
    deterministic: bool
    extrinsic_gamma: float
    extrinsic_strength: int
    init_path: Optional[str]
    keep_checkpoints: int
    even_checkpoints: bool

    # Training Configuration
    max_steps: int
    time_horizon: int
    summary_freq: int
    threaded: bool
    self_play: Optional[str]
    behavioral_cloning: Optional[str]

    # Target
    time_start: int
    time_end: float
    total_steps: int
    total_duration: float
