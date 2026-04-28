CAPABILITY_MATRIX = {
    "fixed": {
        "parse": {"supported": True, "reason": ""},
        "metadata": {"supported": True, "reason": ""},
        "rnea": {"supported": True, "reason": ""},
        "minv": {"supported": True, "reason": ""},
    },
    "floating": {
        "parse": {"supported": True, "reason": ""},
        "metadata": {"supported": True, "reason": ""},
        "rnea": {
            "supported": False,
            "reason": (
                "Floating-base numerical equivalence is not yet trusted because the "
                "Pinocchio free-flyer velocity convention has not been verified "
                "against GRiD's current floating-base representation in this checkout."
            ),
        },
        "minv": {
            "supported": False,
            "reason": (
                "Floating-base inverse-mass equivalence is blocked on the same "
                "free-flyer convention verification as floating-base RNEA."
            ),
        },
    },
}


def get_capability(base_mode: str, algorithm: str):
    return CAPABILITY_MATRIX[base_mode][algorithm]
