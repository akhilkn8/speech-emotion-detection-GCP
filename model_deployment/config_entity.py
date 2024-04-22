from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeploymentConfig:
    """
    Represents the configuration for deployment a machine learning model.

    Attributes:
        model_name: Name of the machine learning model.
        project_id: ID of the project where the model is deployed.
        location: Location where the model is deployed.
        endpoint_id: ID of the endpoint (optional).
        machine_type: Type of machine for deployment.
        min_replica_count: Minimum number of replicas for deployment.
        max_replica_count: Maximum number of replicas for deployment.
        traffic_percentage: Percentage of traffic to allocate to the model.
    """

    model_name: str
    machine_type: str
    min_replica_count: int
    max_replica_count: int
    traffic_percentage: float
    scaler_path: str
    