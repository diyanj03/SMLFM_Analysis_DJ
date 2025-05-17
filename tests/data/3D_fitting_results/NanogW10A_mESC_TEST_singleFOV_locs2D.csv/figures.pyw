#!/usr/bin/env python3
# # # GENERATED FILE # # #
from pathlib import Path
from matplotlib import pyplot as plt
import smlfm
if __name__ == "__main__":
    locs_3d = Path(__file__).parent.absolute() / "locs3D.csv"
    max_lat_err = 200.0
    min_views = 3
    fig1, fig2, fig3 = smlfm.graphs.reconstruct_results(
        plt.figure(), plt.figure(), plt.figure(),
        locs_3d, max_lat_err, min_views)
    fig1.canvas.manager.set_window_title("Occurrences")
    fig2.canvas.manager.set_window_title("Histogram")
    fig3.canvas.manager.set_window_title("3D")
    plt.show()
