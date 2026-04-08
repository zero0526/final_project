from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, Any
import os
import json
import yaml
from pathlib import Path

# Determine the project root (3 levels up from src/configs/configs.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

def get_env_file() -> str:
    if "ENV_FILE" in os.environ:
        return os.environ["ENV_FILE"]
    return str(PROJECT_ROOT / "dev.env")

def load_nodes_types(path_cfg: str)->Dict[str,str]:
    with open(path_cfg, "r", encoding="utf-8") as f:
        data= yaml.safe_load(f)
        data=data["node-cfg"]
        rs={}
        for n, c in data.items():
            rs[n]= {i:t for t, nids in c.items() for i in nids}
        return rs

def load_nodes_config(path_cfg: str):
    with open(path_cfg, "r", encoding="utf-8") as f:
        data= yaml.safe_load(f)
        return data["nodes"]

def default_topology_config(topology: str, config: Any)->Dict[str, Any]:
    from src.utils.CreateSpec import SNDLibLoad

    path_json= str(PROJECT_ROOT / "data" / f"{topology}_nodes_config.json")
    path_xml = str(PROJECT_ROOT / "data" / "topologyXML" / f"{topology}.xml")
    if not os.path.exists(path_json):
        sndLoad= SNDLibLoad(path_xml, config)
        return sndLoad.load()
    else:
        with open(path_json, "r", encoding="utf-8") as f:
            return json.load(f)

def load_neuron_cfg(path_cfg: str):
    with open(path_cfg, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("NEURON_NET")

def load_services(path_cfg: str):
    with open(path_cfg, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("services")

class BaseConfig(BaseSettings):
    topology: str= Field(default="atlanta")
    device: str= Field(default="cpu")
    logs: str= Field(default=str(PROJECT_ROOT / "data" / "logs"))
    checkpoints: str= Field(default=str(PROJECT_ROOT / "data" / "checkpoints"))
    results: str= Field(default=str(PROJECT_ROOT / "data" / "results"))
    node_type_path: str= Field(default=str(PROJECT_ROOT / "data" / "distribute_node" / "nodes.yaml"))
    node_config_path: str= Field(default=str(PROJECT_ROOT / "data"  /"distribute_node" / "node_spec.yaml"))
    neural_cfg_path: str= Field(default=str(PROJECT_ROOT / "data" / "training_cfg.yaml"))
    service_path: str= Field(default=str(PROJECT_ROOT / "data" / "ai_services.yaml"))
    nodes_type:Dict[str,str] = Field(default={})
    nodes_config:Dict[str,str] = Field(default={})
    energy_coef: float = Field(default=5e-10)
    transmission_rate: Dict[str, float] = Field(default={"min": 50, "max": 100})
    topology_data: Dict[str, Any] = Field(default_factory=dict)
    cold_start_energy: float= Field(default= 0.2)
    transmission_coef: float= Field(default=0.2)
    lypa_coef: float= Field(default=1e5)
    cold_start_time: Dict[str, float]= Field(default={"min":0.15, "max":0.85})
    avg_req: int= Field(default=20)
    neighbor_depth: int= Field(default=2)
    task_arrival_rate: float= Field(default=1)
    zipf_param: float= Field(default=0.8)
    default_batch_size: int= Field(default=20)
    hyper_neural: Dict[str, Any]= Field(default={})
    services: Dict[str,str]= Field(default={})

    class Config:
        env_file = get_env_file()
        env_file_encoding = "utf-8"

cfg = BaseConfig()
cfg.nodes_type= load_nodes_types(cfg.node_type_path)
cfg.nodes_config= load_nodes_config(cfg.node_config_path)
cfg.topology_data = default_topology_config(cfg.topology, cfg)
cfg.hyper_neural= load_neuron_cfg(cfg.neural_cfg_path)
cfg.services= load_services(cfg.service_path)



