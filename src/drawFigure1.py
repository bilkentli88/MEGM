import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

# --- CONFIGURATION FOR JOURNAL QUALITY ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['pdf.fonttype'] = 42


def draw_rounded_box(ax, xy, width, height, text, color='#FFFFFF',
                     edgecolor='#B0B0B0', textcolor='#2D2D2D', shadow=True):
    """Draws a professional rounded box with optional drop shadow."""
    x, y = xy

    # 1. Subtle Drop Shadow (lighter and softer for elegance)
    if shadow:
        shadow_rect = patches.FancyBboxPatch(
            (x + 0.004, y - 0.006), width, height,
            boxstyle="round,pad=0,rounding_size=0.02",
            ec="none", fc='#000000', alpha=0.10, zorder=2
        )
        ax.add_patch(shadow_rect)

    # 2. Main Box
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0,rounding_size=0.02",
        ec=edgecolor, fc=color, linewidth=0.8, zorder=3
    )
    ax.add_patch(rect)

    # 3. Text (Dark Grey instead of pure black for elegance)
    ax.text(x + width / 2, y + height / 2, text,
            ha='center', va='center', fontsize=10,
            color=textcolor, fontweight='bold', zorder=4)
    return rect


def draw_arrow(ax, start, end, style='->'):
    """Draws a clean, thin, elegant arrow."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color='#555555', lw=1.0, shrinkA=0, shrinkB=0),
                zorder=1)


def draw_ortho_arrow(ax, start_xy, end_xy, mid_y=None):
    """Draws an L-shaped arrow."""
    if mid_y is None:
        mid_y = (start_xy[1] + end_xy[1]) / 2

    # Draw path
    ax.plot([start_xy[0], start_xy[0], end_xy[0], end_xy[0]],
            [start_xy[1], mid_y, mid_y, end_xy[1]],
            color='#555555', lw=1.0, zorder=1)

    # Arrow head
    ax.annotate('', xy=end_xy, xytext=(end_xy[0], end_xy[1] + 0.001),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0), zorder=1)


def plot_framework_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.75)
    ax.axis('off')

    # --- DEFINITIONS ---
    w_box = 0.18
    h_box = 0.08
    y_top = 0.65
    y_mid = 0.45
    y_grid_top = 0.30
    y_grid_bot = 0.18
    y_res = 0.05

    # --- ELEGANT COLOR PALETTE ---
    # Data: "Cool Slate" (Very subtle blue-grey) - Cleaner than Baby Blue
    c_data = '#F4F6F8'
    ec_data = '#BCCCDC'

    # Process: Pure White - High contrast against data
    c_proc = '#FFFFFF'
    ec_proc = '#B0B0B0'

    # Evaluation: "Soft Mint/Sage" - Professional and calm
    c_eval = '#F0F7F4'
    ec_eval = '#88B3A6'

    # Result: "Refined Cream/Gold" - Stands out but isn't neon
    c_res = '#FFF9E6'
    ec_res = '#E6C200'

    # Text Color: "Charcoal" (Softer than pure black)
    txt_col = '#222222'

    # --- TOP ROW (DATA FLOW) ---
    draw_rounded_box(ax, (0.1, y_top), w_box, h_box, "Real Data\n$\mathcal{D}_r \sim P_r$",
                     color=c_data, edgecolor=ec_data, textcolor=txt_col)

    draw_rounded_box(ax, (0.4, y_top), w_box, h_box, "Generator\n$G_{\\theta}$",
                     color=c_proc, edgecolor=ec_proc, textcolor=txt_col)

    draw_rounded_box(ax, (0.7, y_top), w_box, h_box, "Synthetic Data\n$\mathcal{D}_g \sim P_g$",
                     color=c_data, edgecolor=ec_data, textcolor=txt_col)

    # Top Arrows
    draw_arrow(ax, (0.28, y_top + h_box / 2), (0.4, y_top + h_box / 2))
    draw_arrow(ax, (0.58, y_top + h_box / 2), (0.7, y_top + h_box / 2))

    # --- MIDDLE (EVALUATION) ---
    draw_rounded_box(ax, (0.25, y_mid), 0.5, h_box, "Trust-Oriented Evaluation",
                     color=c_eval, edgecolor=ec_eval, textcolor=txt_col)

    # Routing Arrows (Real/Synth -> Eval) - Dotted lines for logic flow
    ax.plot([0.19, 0.19, 0.25], [y_top, y_mid + h_box / 2, y_mid + h_box / 2], ':', color='#888888', lw=1)
    ax.plot([0.79, 0.79, 0.75], [y_top, y_mid + h_box / 2, y_mid + h_box / 2], ':', color='#888888', lw=1)
    draw_arrow(ax, (0.24, y_mid + h_box / 2), (0.25, y_mid + h_box / 2))
    draw_arrow(ax, (0.76, y_mid + h_box / 2), (0.75, y_mid + h_box / 2))

    # --- GRID OF DIMENSIONS ---
    # Top Row
    draw_rounded_box(ax, (0.1, y_grid_top), w_box, h_box, "Fidelity",
                     color=c_eval, edgecolor=ec_eval, textcolor=txt_col)
    draw_rounded_box(ax, (0.4, y_grid_top), w_box, h_box, "Diversity\n& Coverage",
                     color=c_eval, edgecolor=ec_eval, textcolor=txt_col)
    draw_rounded_box(ax, (0.7, y_grid_top), w_box, h_box, "Fairness\n(Bias)",
                     color=c_eval, edgecolor=ec_eval, textcolor=txt_col)

    # Bottom Row
    draw_rounded_box(ax, (0.1, y_grid_bot), w_box, h_box, "Robustness\n& Stability",
                     color=c_eval, edgecolor=ec_eval, textcolor=txt_col)
    draw_rounded_box(ax, (0.4, y_grid_bot), w_box, h_box, "Utility",
                     color=c_eval, edgecolor=ec_eval, textcolor=txt_col)
    draw_rounded_box(ax, (0.7, y_grid_bot), w_box, h_box, "Safety\n(Optional)",
                     color=c_eval, edgecolor=ec_eval, textcolor=txt_col)

    # Arrows from Eval to Dimensions
    draw_arrow(ax, (0.5, y_mid), (0.5, y_grid_top + h_box), style='->')
    ax.plot([0.5, 0.19, 0.19], [y_mid, y_mid, y_grid_top + h_box], color='#555555', lw=1.0)  # To Fidelity
    draw_arrow(ax, (0.19, y_grid_top + h_box + 0.005), (0.19, y_grid_top + h_box))
    ax.plot([0.5, 0.79, 0.79], [y_mid, y_mid, y_grid_top + h_box], color='#555555', lw=1.0)  # To Fairness
    draw_arrow(ax, (0.79, y_grid_top + h_box + 0.005), (0.79, y_grid_top + h_box))

    # Arrows between dimensions
    draw_arrow(ax, (0.19, y_grid_top), (0.19, y_grid_bot + h_box))
    draw_arrow(ax, (0.49, y_grid_top), (0.49, y_grid_bot + h_box))
    draw_arrow(ax, (0.79, y_grid_top), (0.79, y_grid_bot + h_box))

    # --- BOTTOM (RESULT) ---
    draw_rounded_box(ax, (0.25, y_res), 0.5, h_box, "Trust Profile\n$\\mathcal{T}(G_{\\theta})$",
                     color=c_res, edgecolor=ec_res, textcolor=txt_col)

    # Final Aggregation Arrows
    draw_ortho_arrow(ax, (0.19, y_grid_bot), (0.25, y_res + h_box / 2), mid_y=y_grid_bot - 0.02)
    draw_arrow(ax, (0.49, y_grid_bot), (0.49, y_res + h_box))
    draw_ortho_arrow(ax, (0.79, y_grid_bot), (0.75, y_res + h_box / 2), mid_y=y_grid_bot - 0.02)

    # Save
    print("Saving Figure1_Framework_Elegant.pdf...")
    plt.savefig("Figure1_Framework_Elegant.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("Figure1_Framework_Elegant.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_framework_diagram()