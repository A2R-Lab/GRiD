CAPABILITY_MATRIX = {
    "fixed": {
        "parse": {"supported": True, "reason": ""},
        "metadata": {"supported": True, "reason": ""},
        "rnea": {"supported": True, "reason": ""},
        "minv": {"supported": True, "reason": ""},
        "forward_dynamics": {"supported": True, "reason": ""},
        "rnea_grad": {"supported": True, "reason": ""},
        "forward_dynamics_grad": {"supported": True, "reason": ""},
    },
    "floating": {
        "parse": {"supported": True, "reason": ""},
        "metadata": {"supported": True, "reason": ""},
        "rnea": {
            "supported": True,
            "reason": "",
        },
        "minv": {
            "supported": True,
            "reason": "",
        },
        "forward_dynamics": {
            "supported": True,
            "reason": "",
        },
        "rnea_grad": {
            "supported": True,
            "reason": "",
        },
        "forward_dynamics_grad": {
            "supported": True,
            "reason": "",
        },
    },
}


def get_capability(base_mode: str, algorithm: str):
    return CAPABILITY_MATRIX[base_mode][algorithm]
