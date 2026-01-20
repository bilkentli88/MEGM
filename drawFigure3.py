import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CONFIGURATION ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['pdf.fonttype'] = 42


def draw_box(ax, xy, width, height, title, content,
             style='round', color='#FFFFFF', edgecolor='#B0B0B0', textcolor='#2D2D2D'):
    """Draws a box with distinct styling for Title (bold) and Content (normal)."""
    x, y = xy

    # --- STYLE LOGIC ---
    # Standard Step Box
    b_style = "round,pad=0.01,rounding_size=0.02"

    # Start/End Node (Rounder pill shape)
    if style == 'pill':
        b_style = "round,pad=0.01,rounding_size=0.08"

        # 1. Drop Shadow
    shadow_rect = patches.FancyBboxPatch(
        (x + 0.005, y - 0.008), width, height,
        boxstyle=b_style,
        ec="none", fc='#000000', alpha=0.10, zorder=2
    )
    ax.add_patch(shadow_rect)

    # 2. Main Box
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle=b_style,
        ec=edgecolor, fc=color, linewidth=0.8, zorder=3
    )
    ax.add_patch(rect)

    # 3. Text Positioning
    cx = x + width / 2

    if content:
        # Case A: Title AND Content (Steps 1-3, and now Output)
        # Title: Bold, positioned near top
        title_y = y + (height * 0.78)
        ax.text(cx, title_y, title,
                ha='center', va='center', fontsize=11,
                color=textcolor, fontweight='bold', zorder=4)

        # Content: Normal weight, smaller size, positioned in lower half
        content_y = y + (height * 0.35)
        ax.text(cx, content_y, content,
                ha='center', va='center', fontsize=10,
                color=textcolor, linespacing=1.5, zorder=4)
    else:
        # Case B: Title Only (Input Node)
        # Centered, Bold
        cy = y + height / 2
        ax.text(cx, cy, title,
                ha='center', va='center', fontsize=11,
                color=textcolor, fontweight='bold', zorder=4)


def draw_arrow(ax, start_xy, end_xy):
    """Draws a clean vertical arrow."""
    ax.annotate('', xy=end_xy, xytext=start_xy,
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2, shrinkA=0, shrinkB=0),
                zorder=1)


def plot_protocol_diagram_final_v2():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # --- PALETTE ---
    c_input = '#F4F6F8'  # Cool Slate
    c_step = '#FFFFFF'  # White
    c_out = '#FFF9E6'  # Cream/Gold
    ec_step = '#B0B0B0'
    ec_input = '#BCCCDC'
    ec_out = '#E6C200'
    txt_col = '#222222'

    # --- GEOMETRY ---
    w_box = 0.6
    h_box = 0.15
    h_cloud = 0.08
    h_out = 0.10  # Slightly taller than input to fit two lines nicely
    x_center = 0.2

    # Vertical spacing
    y_input = 0.90
    y_step1 = 0.71
    y_step2 = 0.51
    y_step3 = 0.31
    y_out = 0.13

    # 1. INPUT (Start Node) - Title only
    draw_box(ax, (x_center, y_input), w_box, h_cloud,
             "Input Models $\mathcal{M}=\{G_1, \dots, G_K\}$", None,
             style='pill', color=c_input, edgecolor=ec_input, textcolor=txt_col)

    # 2. STEP 1
    content1 = "Compute raw scalar values for applicable dimensions:\nFidelity, Diversity, Bias, Robustness, Utility, Safety"
    draw_box(ax, (x_center, y_step1), w_box, h_box,
             "Step 1: Metric Computation", content1,
             style='round', color=c_step, edgecolor=ec_step, textcolor=txt_col)

    # 3. STEP 2
    content2 = "Map raw metric values $v_k^{(i)}$ to $[0,1]$\nusing min–max scaling relative to set $\mathcal{M}$"
    draw_box(ax, (x_center, y_step2), w_box, h_box,
             "Step 2: Normalization", content2,
             style='round', color=c_step, edgecolor=ec_step, textcolor=txt_col)

    # 4. STEP 3
    content3 = "Construct Trust Profile:\n$\\mathcal{T}(G_i) = [\\hat{T}_{\\mathrm{fid}}^{(i)}, \\hat{T}_{\\mathrm{div}}^{(i)}, \\dots, \\hat{T}_{\\mathrm{safe}}^{(i)}]$"
    draw_box(ax, (x_center, y_step3), w_box, h_box,
             "Step 3: Profile Construction", content3,
             style='round', color=c_step, edgecolor=ec_step, textcolor=txt_col)

    # 5. OUTPUT (End Node) - Split Title and Content
    title_out = "Visualization and Analysis"
    content_out = "Radar plots and trade-off inspection"
    draw_box(ax, (x_center, y_out), w_box, h_out,
             title_out, content_out,
             style='pill', color=c_out, edgecolor=ec_out, textcolor=txt_col)

    # --- ARROWS ---
    xc = 0.5
    draw_arrow(ax, (xc, y_input), (xc, y_step1 + h_box))
    draw_arrow(ax, (xc, y_step1), (xc, y_step2 + h_box))
    draw_arrow(ax, (xc, y_step2), (xc, y_step3 + h_box))
    draw_arrow(ax, (xc, y_step3), (xc, y_out + h_out))

    print("Saving Figure3_Protocol_Final_v2.pdf...")
    plt.savefig("Figure3_Protocol_Final_v2.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("Figure3_Protocol_Final_v2.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_protocol_diagram_final_v2()