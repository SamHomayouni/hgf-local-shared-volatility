"""
Global-only pyhgf model.

Architecture
------------
Two observation streams, A and B.

Each stream has:
    - its own input node
    - its own value/state node

Both value/state nodes share one volatility parent.

Conceptual structure
--------------------
    input_A  <- value_A  <-
                          shared_volatility
    input_B  <- value_B  <-

Expected fingerprint
--------------------
If stream A becomes volatile:
    - learning rate for A should increase
    - learning rate for B may also increase because the volatility estimate is shared
"""

from __future__ import annotations
from pyhgf.model import Network
MODEL_NAME = "global_only"
NODE_IDS = {
    "input_A": 0,
    "input_B": 1,
    "value_A": 2,
    "value_B": 3,
    "shared_volatility": 4,
}

def get_node_ids() -> dict[str, int]:
    """
    Return node IDs for this model.
    """
    return dict(NODE_IDS)

def build_global_only_model(*, input_precision: float = 100.0,
                            value_mean: float = 0.0,
                            value_precision: float = 1.0,
                            value_tonic_volatility: float = -2.0,
                            value_autoconnection_strength: float = 1, #0.95
                            shared_volatility_mean: float = 0.0,
                            shared_volatility_precision: float = 1.0,
                            shared_volatility_tonic_volatility: float = -3.0, #-6.0
                            value_coupling: float = 1.0,
                            shared_volatility_coupling: float = 1.0,
                            update_type: str = "eHGF",
                            max_posterior_precision: float = 1e10) -> Network:
    """
    Build the global-only model.,

    Parameters
    ----------
    input_precision:
        Precision of the continuous input nodes.

    value_mean:
        Initial mean of the value/state nodes.

    value_precision:
        Initial precision of the value/state nodes.

    value_tonic_volatility:
        Baseline log-volatility of the value/state nodes.

    shared_volatility_mean:
        Initial mean of the shared volatility parent.

    shared_volatility_precision:
        Initial precision of the shared volatility parent.

    shared_volatility_tonic_volatility:
        Baseline log-volatility of the shared volatility parent.

    value_coupling:
        Coupling strength from value nodes to input nodes.

    shared_volatility_coupling:
        Coupling strength from shared_volatility to both value nodes.

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
        raise ValueError("value_autoconnection_strength must be between 0 and 1")
    

    model = Network(update_type= update_type, max_posterior_precision= max_posterior_precision)
    #node 0 and 1: continouos input nodes for stream A and B
    model = model.add_nodes(n_nodes= 2, precision = input_precision)
    #node 2: value_state node predicting input_A(node 0)
    model = model.add_nodes(value_children=([NODE_IDS["input_A"]], [value_coupling]),
                            mean = value_mean,
                            precision = value_precision,
                            tonic_volatility = value_tonic_volatility,
                            autoconnection_strength = value_autoconnection_strength)
    #node 3: value_state node predicting input_B(node 1)
    model = model.add_nodes(value_children=([NODE_IDS["input_B"]], [value_coupling]),
                            mean = value_mean,
                            precision = value_precision,
                            tonic_volatility= value_tonic_volatility,
                            autoconnection_strength = value_autoconnection_strength)
    #node 4: shared volatility parent for value_A and value_B(nodes 2 and 3)
    model = model.add_nodes(volatility_children=([NODE_IDS["value_A"], NODE_IDS["value_B"]], [shared_volatility_coupling, shared_volatility_coupling]),
                            mean = shared_volatility_mean,
                            precision = shared_volatility_precision,
                            tonic_volatility = shared_volatility_tonic_volatility)
    return model

def build_model(**kwargs) -> Network:
    """
    Standard alias used by runner scripts.
    """
    return build_global_only_model(**kwargs)