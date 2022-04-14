class MissingVariableError(Exception):
    def __init__(self, metric_name: str, *vars) -> None:
        """Missing necessary variables for computing the metric"""

        msg = f"Missing required variables for computing {metric_name}: {', '.join(*vars)}."
        super().__init__(msg)
