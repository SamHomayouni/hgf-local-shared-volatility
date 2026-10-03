"""
Local-only pyhgf model.

Architecture
------------
Two observation streams, A and B.

Each stream has:
    - its own input node
    - its own value/state node
    - its own volatility parent

There is no shared/global volatility node.

Conceptual structure
--------------------
    input_A  <- value_A  <- volatility_A
    input_B  <- value_B  <- volatility_B

Expected fingerprint
--------------------
If stream A becomes volatile:
    - learning rate for A should increase
    - learning rate for B should not systematically increase
"""

from __future__ import annotations
from pyhgf.model import Network

MODEL_NAME = "local_only"
NODE_IDS = {
    "input_A": 0,
    "input_B": 1,
    "value_A": 2,
    "value_B": 3,
    "volatility_A": 4,
    "volatility_B": 5,
}

def get_node_ids() -> dict[str, int]:
    """
    Return node IDs for this model.

    pyhgf uses integer node indices internally. Keeping the mapping here makes
    later extraction of trajectories less error-prone.
    """
    return dict(NODE_IDS)

def build_local_only_model(*, input_precision: float = 100.0,
                           value_mean: float = 0.0,
                           value_precision: float = 1.0,
                           value_tonic_volatility: float = -7.824,
                           value_autoconnection_strength: float = 0.95,
                           volatility_mean: float = 0.0,
                           volatility_precision: float = 1.0,
                           volatility_tonic_volatility: float = -6.0,
                           value_coupling: float = 1.0,
                           volatility_coupling: float = 1.0,
                           update_type: str = "eHGF",
                           max_posterior_precision: float = 1e10) -> Network:
    """
    Build the local-only model.

    Parameters
    ----------
    input_precision:
        Precision of the continuous input nodes. If your generated observation
        noise has sd = 0.10, then precision = 1 / 0.10**2 = 100.

    value_mean:
        Initial mean of the value/state nodes.

    value_precision:
        Initial precision of the value/state nodes.

    value_tonic_volatility:
        Baseline log-volatility of the value/state nodes.

    volatility_mean:
        Initial mean of each local volatility parent.

    volatility_precision:
        Initial precision of each local volatility parent.

    volatility_tonic_volatility:
        Baseline log-volatility of each volatility parent.

    value_coupling:
        Coupling strength from value_A to input_A and value_B to input_B.

    volatility_coupling:
        Coupling strength from volatility_A to value_A and volatility_B to value_B.

    update_type:
        pyhgf volatility-update type. Default is "eHGF".

    max_posterior_precision:
        Upper bound for posterior precision writes.

    Returns
    -------
    pyhgf.model.Network
        Unfitted pyhgf network.
    """
    if not 0.0 <= value_autoconnection_strength <= 1.0:
        raise ValueError("value_autoconnection_strength must be betweem 0 and 1")
    

    model = Network(update_type= update_type, max_posterior_precision= max_posterior_precision)
    #node 0 and 1: continouos input nodes for stream A and stream B.
    model = model.add_nodes(n_nodes = 2, precision = input_precision)
    #node 2: value_state node predicting input_A
    model = model.add_nodes(value_children = ([NODE_IDS["input_A"]], [value_coupling]),
                            mean = value_mean,
                            precision = value_precision,
                            tonic_volatility = value_tonic_volatility,
                            autoconnection_strength = value_autoconnection_strength)
    #node 3: value_state node predicting input_B
    model = model.add_nodes(value_children = ([NODE_IDS["input_B"]], [value_coupling]),
                            mean = value_mean,
                            precision = value_precision,
                            tonic_volatility = value_tonic_volatility,
                            autoconnection_strength = value_autoconnection_strength)
    #node 4: local volatility parent for value_A(node 2)
    model = model.add_nodes(volatility_children =([NODE_IDS["value_A"]], [volatility_coupling]),
                            mean = volatility_mean,
                            precision = volatility_precision,
                            tonic_volatility = volatility_tonic_volatility)
    #node 5: local volatility parent for value_B(node 3)
    model = model.add_nodes(volatility_children = ([NODE_IDS["value_B"]], [volatility_coupling]),
                            mean = volatility_mean,
                            precision = volatility_precision,
                            tonic_volatility = volatility_tonic_volatility)
    
    return model

def build_model(**kwargs) -> Network:
    """
    Standard alias used by runner scripts.
    """
    return build_local_only_model(**kwargs)