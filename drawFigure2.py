import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CONFIGURATION ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['pdf.fonttype'] = 42


def draw_rounded_box(ax, xy, width, height, title, bullets,
                     color='#FFFFFF', edgecolor='#B0B0B0', textcolor='#2D2D2D'):
    """Draws a box with a centered Title but LEFT-ALIGNED bullets."""
    x, y = xy

    # 1. Drop Shadow
    shadow_rect = patches.FancyBboxPatch(
        (x + 0.005, y - 0.008), width, height,
        boxstyle="round,pad=0,rounding_size=0.03",
        ec="none", fc='#000000', alpha=0.10, zorder=2
    )
    ax.add_patch(shadow_rect)

    # 2. Main Box
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0,rounding_size=0.03",
        ec=edgecolor, fc=color, linewidth=0.8, zorder=3
    )
    ax.add_patch(rect)

    # 3. Title (Centered)
    ax.text(x + width / 2, y + height - 0.04, title,
            ha='center', va='top', fontsize=11,
            color=textcolor, fontweight='bold', zorder=4)

    # 4. Bullets (Left-Aligned Logic)
    bullet_text = "\n".join([f"• {b}" for b in bullets])
    # Fixed left margin relative to box for perfect left alignment
    left_margin = x + 0.03

    ax.text(left_margin, y + height / 2 - 0.02, bullet_text,
            ha='left', va='center', fontsize=10,
            color=textcolor, linespacing=1.6, zorder=4)

    return rect


def draw_root_box(ax, xy, width, height, text,
                  color='#FFFFFF', edgecolor='#B0B0B0', textcolor='#2D2D2D'):
    """Draws the single Root node."""
    x, y = xy
    # Shadow
    ax.add_patch(patches.FancyBboxPatch(
        (x + 0.005, y - 0.008), width, height,
        boxstyle="round,pad=0,rounding_size=0.03",
        ec="none", fc='#000000', alpha=0.10, zorder=2
    ))
    # Box
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0,rounding_size=0.03",
        ec=edgecolor, fc=color, linewidth=1.0, zorder=3
    ))
    # Text
    ax.text(x + width / 2, y + height / 2, text,
            ha='center', va='center', fontsize=12,
            color=textcolor, fontweight='bold', zorder=4)


def draw_arrow(ax, start_xy, end_xy):
    """Draws a clean arrow."""
    ax.annotate('', xy=end_xy, xytext=start_xy,
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2, shrinkA=0, shrinkB=0),
                zorder=1)


def plot_taxonomy_diagram_final():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # --- PALETTE ---
    c_box = '#F0F7F4'  # Soft Mint/Sage
    ec_box = '#88B3A6'  # Teal border
    c_root = '#E0F2F1'
    ec_root = '#00897B'
    txt_col = '#222222'

    # --- GEOMETRY ---
    w_box = 0.26
    h_box = 0.16
    y_root = 0.82
    y_row1 = 0.52
    y_row2 = 0.15
    x_left = 0.08
    x_mid = 0.37
    x_right = 0.66

    # 1. ROOT
    draw_root_box(ax, (0.30, y_root), 0.40, 0.10, "Trust-Oriented Evaluation Metrics",
                  color=c_root, edgecolor=ec_root, textcolor=txt_col)

    # 2. ROW 1
    draw_rounded_box(ax, (x_left, y_row1), w_box, h_box, "Fidelity",
                     ["Distributional distance", "Feature-space similarity"],
                     color=c_box, edgecolor=ec_box, textcolor=txt_col)

    draw_rounded_box(ax, (x_mid, y_row1), w_box, h_box, "Diversity / Coverage",
                     ["Dispersion", "Mode coverage"],
                     color=c_box, edgecolor=ec_box, textcolor=txt_col)

    draw_rounded_box(ax, (x_right, y_row1), w_box, h_box, "Bias / Fairness",
                     ["Marginal distortion", "Conditional disparity"],
                     color=c_box, edgecolor=ec_box, textcolor=txt_col)

    # 3. ROW 2
    draw_rounded_box(ax, (x_left, y_row2), w_box, h_box, "Robustness / Stability",
                     ["Latent sensitivity", "Run-to-run variance"],
                     color=c_box, edgecolor=ec_box, textcolor=txt_col)

    draw_rounded_box(ax, (x_mid, y_row2), w_box, h_box, "Utility",
                     ["Downstream performance"],
                     color=c_box, edgecolor=ec_box, textcolor=txt_col)

    draw_rounded_box(ax, (x_right, y_row2), w_box, h_box, "Safety (Optional)",
                     ["Constraint satisfaction", "Valid generations"],
                     color=c_box, edgecolor=ec_box, textcolor=txt_col)

    # --- ARROWS (CORRECTED) ---
    branch_y = (y_root + (y_row1 + h_box)) / 2 + 0.02

    # 1. Main stem down from Root
    ax.plot([0.5, 0.5], [y_root, branch_y], color='#555555', lw=1.2)

    # 2. Branches to Row 1 (Right-Angle Connectors)

    # To Fidelity (Left) - Horizontal bar then vertical arrow
    ax.plot([0.5, x_left + w_box / 2], [branch_y, branch_y], color='#555555', lw=1.2)
    draw_arrow(ax, (x_left + w_box / 2, branch_y), (x_left + w_box / 2, y_row1 + h_box))

    # To Diversity (Middle) - Straight down arrow
    draw_arrow(ax, (0.5, branch_y), (x_mid + w_box / 2, y_row1 + h_box))

    # To Bias (Right) - Horizontal bar then vertical arrow
    ax.plot([0.5, x_right + w_box / 2], [branch_y, branch_y], color='#555555', lw=1.2)
    draw_arrow(ax, (x_right + w_box / 2, branch_y), (x_right + w_box / 2, y_row1 + h_box))

    # 3. Vertical connectors from Row 1 to Row 2
    draw_arrow(ax, (x_left + w_box / 2, y_row1), (x_left + w_box / 2, y_row2 + h_box))
    draw_arrow(ax, (x_mid + w_box / 2, y_row1), (x_mid + w_box / 2, y_row2 + h_box))
    draw_arrow(ax, (x_right + w_box / 2, y_row1), (x_right + w_box / 2, y_row2 + h_box))

    # Save
    print("Saving Figure2_Taxonomy_Final.pdf...")
    plt.savefig("Figure2_Taxonomy_Final.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("Figure2_Taxonomy_Final.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_taxonomy_diagram_final()