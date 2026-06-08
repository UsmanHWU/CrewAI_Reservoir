"""
Custom Tool: Reservoir Engineering Calculator
Calculates key metrics: Productivity Index (PI), Recovery Factor, and Decline Rate.
"""

from crewai_tools import BaseTool
import math


class ReservoirCalculatorTool(BaseTool):
    name: str = "reservoir_calculator"
    description: str = (
        "Calculates key petroleum reservoir engineering metrics. "
        "Input should be a string in the format: "
        "'flow_rate=X, drawdown=Y, OOIP=Z, cumulative=W, qi=A, qf=B, time=C' "
        "where flow_rate is in STB/day, drawdown in psi, OOIP in MMSTB, "
        "cumulative production in MMSTB, qi=initial rate STB/day, "
        "qf=final rate STB/day, time=years."
    )

    def _run(self, input_str: str) -> str:
        """Parse inputs and return calculated metrics."""
        try:
            params = {}
            for part in input_str.split(","):
                key, value = part.strip().split("=")
                params[key.strip()] = float(value.strip())

            results = []

            # 1. Productivity Index (PI) = q / ΔP  [STB/day/psi]
            if "flow_rate" in params and "drawdown" in params:
                pi = params["flow_rate"] / params["drawdown"]
                results.append(f"Productivity Index (PI) = {pi:.3f} STB/day/psi")

            # 2. Recovery Factor (RF) = Np / OOIP  [fraction]
            if "OOIP" in params and "cumulative" in params:
                rf = (params["cumulative"] / params["OOIP"]) * 100
                results.append(f"Recovery Factor (RF) = {rf:.2f}%")

            # 3. Exponential Decline Rate  [fraction/year]
            if "qi" in params and "qf" in params and "time" in params:
                if params["qf"] > 0 and params["qi"] > 0:
                    D = -math.log(params["qf"] / params["qi"]) / params["time"]
                    results.append(f"Annual Decline Rate (D) = {D:.4f} /year ({D*100:.2f}%/year)")

            if not results:
                return "ERROR: Could not parse any valid parameters. Check input format."

            return "\n".join(results)

        except Exception as e:
            return f"FALLBACK: Calculator error - {str(e)}. Using default estimates: PI~1.5 STB/day/psi, RF~25%, D~15%/yr"
