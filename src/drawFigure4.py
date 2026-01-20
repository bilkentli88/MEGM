import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_var_labels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return Circle((0.5, 0.5), 0.5)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def plot_trust_profile():
    # --- 1. DATA (Mean values from Table 4 - Experiment III) ---
    metrics = ['Fidelity', 'Diversity', 'Robustness', 'Utility', 'Fairness', 'Safety']

    # UPDATED DATA: Using the exact values from your new Table 4
    raw_data = {
        'Baseline': [2.464, 18.74, 3.1e-4, 0.758, 0.592, 0.604],
        'Robust':   [1.709, 16.69, 1.6e-5, 0.746, 0.615, 0.689],
        'TimeGAN':  [1.396, 17.17, 5.9e-2, 0.532, 0.706, 0.320],
        'LSTM-VAE': [1.383, 10.33, 3.0e-3, 0.416, 0.742, 0.885]
    }

    # --- 2. NORMALIZATION ---
    # Stack data in order: Baseline, Robust, TimeGAN, LSTM-VAE
    data_matrix = np.array([
        raw_data['Baseline'],
        raw_data['Robust'],
        raw_data['TimeGAN'],
        raw_data['LSTM-VAE']
    ])

    # Indices where "Lower is Better" (Invert these: 1 - norm)
    # Fidelity (idx 0), Robustness (idx 2)
    lower_is_better = [0, 2]

    norm_data = np.zeros_like(data_matrix)
    for i in range(len(metrics)):
        col = data_matrix[:, i]
        min_v = np.min(col)
        max_v = np.max(col)

        # Handle zero variance case
        if np.isclose(max_v, min_v):
            norm_col = np.ones_like(col)
        else:
            norm_col = (col - min_v) / (max_v - min_v)

        if i in lower_is_better:
            norm_col = 1.0 - norm_col  # Invert so "outer" is always better

        norm_data[:, i] = norm_col

    # --- 3. PLOTTING ---
    N = len(metrics)
    theta = radar_factory(N, frame='polygon')

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    # UPDATED COLORS to match the Paper Text:
    colors = {
        'Baseline': '#E69F00',  # Orange
        'Robust':   '#009E73',  # Green
        'TimeGAN':  '#0072B2',  # Teal/Blue (Fixed to match text)
        'LSTM-VAE': '#CC79A7'   # Purple (Fixed to match text)
    }

    styles = {
        'Baseline': ':',    # Dotted
        'Robust':   '-',    # Solid
        'TimeGAN':  '-.',   # Dash-dot
        'LSTM-VAE': '--'    # Dashed
    }

    models = ['Baseline', 'Robust', 'TimeGAN', 'LSTM-VAE']

    for i, model in enumerate(models):
        values = norm_data[i]
        ax.plot(theta, values, color=colors[model], linestyle=styles[model], linewidth=2.5, label=model)
        ax.fill(theta, values, facecolor=colors[model], alpha=0.1)

    # Grid and Labels
    ax.set_var_labels(metrics)

    # --- FIX APPLIED HERE ---
    # 1. Set positions and labels (no styling args)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=[], angle=0)
    # 2. Style the grid lines separately
    ax.yaxis.grid(True, color='#AAAAAA', linestyle='--', linewidth=0.5)

    ax.set_ylim(0, 1.05)

    # Legend & Title
    legend = ax.legend(loc=(0.85, 0.95), labelspacing=0.3, fontsize=11, frameon=True)
    plt.title("Trust Profile Comparison\n(Normalized: Outer is Better)", y=1.08, fontsize=16, fontweight='bold')

    # --- 4. SAVING ---
    print("Saving Figure 4 as PDF and PNG...")
    plt.savefig("Figure4_TrustProfile.pdf", bbox_inches='tight', format='pdf')
    plt.savefig("Figure4_TrustProfile.png", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_trust_profile()