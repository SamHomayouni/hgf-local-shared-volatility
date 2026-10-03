"""
Shared-volatility / hybrid pyhgf model.

Architecture
------------
Two observation streams, A and B.

Each stream has:
    - its own input node
    - its own value/state node
    - its own local volatility parent

The two local volatility parents share one higher value parent.

Conceptual structure
--------------------
    input_A <- value_A <- local_volatility_A <-
                                              shared_volatility
    input_B <- value_B <- local_volatility_B <-

Interpretation
--------------
This model allows both:
    - stream-specific volatility estimation
    - partial sharing through a higher common parent

Expected fingerprint
--------------------
If stream A becomes volatile:
    - learning rate for A should increase
    - learning rate for B may increase partially
"""

from __future__ import annotations
from pyhgf.model import Network

MODEL_NAME = "shared_volatility"

NODE_IDS = {
    "input_A": 0,
    "input_B": 1,
    "value_A": 2,
    "value_B": 3,
    "local_vol_A": 4,
    "local_vol_B": 5,
    "shared_volatility": 6,
}

def get_node_ids() -> dict[str, int]:
    """
    Return node IDs for this model.
    """
    return dict(NODE_IDS)


def build_shared_vol_model(*, input_precision: float = 100.0,   #fixed
                           value_mean: float = 0.0, #fixed
                           value_precision: float = 1.0,    #fixed
                           value_tonic_volatility: float = -7.824, #exp => 0.0004 and fixed
                           value_autoconnection_strength: float = 0.95, #fixed
                           local_vol_mean: float = 0.0,
                           local_vol_precision: float = 1.0,
                           local_vol_tonic_volatility: float = -6.0, #exp(-6) ~ 0.0025
                           shared_value_mean: float = 0.0,
                           shared_value_precision:float = 1.0,
                           shared_value_tonic_volatility: float = -4.0, #changed from -6.0
                           value_coupling: float = 1.0,
                           local_volatility_coupling: float = 0.7,  #alpha changed from 1.0
                           shared_value_coupling: float = 0.75, #changed from 1.0
                           update_type: str = "eHGF",
                           local_vol_autoconnection_strength: float = 0.7,  #lambda
                           max_posterior_precision: float = 1e10) -> Network:
    """
    Build the shared-volatility / hybrid model.

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

    local_vol_mean:
        Initial mean of the local volatility parents.

    local_vol_precision:
        Initial precision of the local volatility parents.

    local_vol_tonic_volatility:
        Baseline log-volatility of the local volatility parents.

    shared_value_mean:
        Initial mean of the shared parent of the local volatility nodes.

    shared_value_precision:
        Initial precision of the shared parent.

    shared_value_tonic_volatility:
        Baseline log-volatility of the shared parent.

    value_coupling:
        Coupling strength from value nodes to input nodes.

    local_volatility_coupling:
        Coupling strength from local volatility nodes to value nodes.

    shared_value_coupling:
        Value-coupling strength from the shared parent to both local volatility nodes.

    update_type:
        pyhgf volatility-update type. Default is "eHGF".

    local_vol_autoconnection_strength:
        Autoconnection strength lambda for local volatility nodes 4 and 5. Values below 1 make these nodes autoregressive/mean-reverting.

    max_posterior_precision:
        Upper bound for posterior precision writes.

    Returns
    -------
    pyhgf.model.Network
        Unfitted pyhgf network.
    """

    if not 0.0 <= value_autoconnection_strength <= 1.0:
        raise ValueError("value_autoconnection_strength must be between 0 and 1")
    

    if not 0.0 <= local_vol_autoconnection_strength <= 1:
        raise ValueError("local_vol_autoconnection_strength must be between 0 and 1")

    model = Network(update_type= update_type, max_posterior_precision= max_posterior_precision)
    #nodes 0 and 1: continouos input nodes for stream A and B
    model = model.add_nodes(n_nodes= 2, precision = input_precision)
    #node 2: value_state node predicting input_A(node 0)
    model = model.add_nodes(value_children= ([NODE_IDS["input_A"]], [value_coupling]),
                            mean = value_mean,
                            precision = value_precision,
                            tonic_volatility = value_tonic_volatility,
                            autoconnection_strength = value_autoconnection_strength)
    #node 3: value_state node predictiong input_B(node 1)
    model = model.add_nodes(value_children= ([NODE_IDS["input_B"]], [value_coupling]),
                            mean = value_mean,
                            precision = value_precision,
                            tonic_volatility = value_tonic_volatility,
                            autoconnection_strength = value_autoconnection_strength)
    #node 4: local volatility parent for value_A(node 2)
    model = model.add_nodes(volatility_children= ([NODE_IDS["value_A"]], [local_volatility_coupling]),
                            mean = local_vol_mean,
                            precision = local_vol_precision,
                            tonic_volatility = local_vol_tonic_volatility,
                            autoconnection_strength = local_vol_autoconnection_strength)
    #node 5: local volatility parent for value_B(node 3)
    model = model.add_nodes(volatility_children= ([NODE_IDS["value_B"]], [local_volatility_coupling]),
                            mean = local_vol_mean,
                            precision = local_vol_precision,
                            tonic_volatility = local_vol_tonic_volatility,
                            autoconnection_strength = local_vol_autoconnection_strength)
    #node 6: shared value parent of local volatility nodes 4 and 5. Through value coupling, it contributes to the predicted mean of the local volatility states.
    model = model.add_nodes(value_children= ([NODE_IDS["local_vol_A"], NODE_IDS["local_vol_B"]], [shared_value_coupling, shared_value_coupling]),
                            mean = shared_value_mean,
                            precision = shared_value_precision,
                            tonic_volatility = shared_value_tonic_volatility)
    return model

def build_model(**kwargs) -> Network:
    """
    Standard alias used by runner scripts.
    """
    return build_shared_vol_model(**kwargs)