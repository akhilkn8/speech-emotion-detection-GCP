from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServingConfig:
    """
    Represents the configuration for serving a machine learning model.

    Attributes:
        model_name: Name of the machine learning model.
        project_id: ID of the project where the model is deployed.
        location: Location where the model is deployed.
        endpoint_id: ID of the endpoint (optional).
        machine_type: Type of machine for serving.
        min_replica_count: Minimum number of replicas for serving.
        max_replica_count: Maximum number of replicas for serving.
        traffic_percentage: Percentage of traffic to allocate to the model.
    """

    model_name: str
    project_id: str
    location: str
    endpoint_id: str = None
    machine_type: str
    min_replica_count: int
    max_replica_count: int
    traffic_percentage: float
