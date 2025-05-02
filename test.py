import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.utils import plot_conf_bands


# Test script with dummy data
if __name__ == "__main__":
    # Create dummy data
    # x values common to all groups (0 to 99)
    x_values = np.arange(100)

    # Create y values with different decay patterns for each group
    y_values_by_group = {
        "poem": [
            # First sample: power law decay
            (x_values + 1) ** -1.0,
            # Second sample: slightly different decay
            (x_values + 1) ** -1.1,
            # Third sample: another variation
            (x_values + 1) ** -0.9,
        ],
        "gsm8k": [
            # First sample: different power law
            (x_values + 1) ** -1.5,
            # Second sample: variation
            (x_values + 1) ** -1.6,
            # Third sample: another variation
            (x_values + 1) ** -1.4,
        ],
        "code": [
            # Exponential decay pattern
            0.9**x_values,
            # Variations
            0.89**x_values,
            0.91**x_values,
        ],
    }

    # Create directory for output
    output_dir = "test_output"
    Path(output_dir).mkdir(exist_ok=True)

    # Call the function with dummy data
    save_path = plot_conf_bands(
        x_values, y_values_by_group, f"{output_dir}/test_confidence_bands.png"
    )

    print(f"Test completed. Plot saved to {save_path}")
