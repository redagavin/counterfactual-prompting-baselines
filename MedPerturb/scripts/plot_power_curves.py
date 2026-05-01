# ABOUTME: Generates power curve figures for main text and appendix
# ABOUTME: Reads simulation CSVs and produces PDF figures with consistent styling

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os

matplotlib.use('Agg')

METRIC_STYLES = {
    'jsd':       ('JSD',       '#1f77b4', '-', 'o', (0, 3)),
    'kl':        ('KL',        '#ff7f0e', '-', 's', (1, 3)),
    'mi':        ('MI',        '#2ca02c', '-', '^', (0, 3)),
    'phi':       (r'$\phi$',   '#d62728', '-', 'v', (1, 3)),
    'flip_rate': ('Flip Rate', '#9467bd', '-', 'D', (2, 3)),
}

LINEWIDTH = 1.2
MARKERSIZE = 2.5


def plot_single_panel(ax, csv_file, sigma, title):
    """Plot power curves for one (condition, sigma) on the given axes."""
    df = pd.read_csv(csv_file)
    df_sig = df[df['sigma'] == sigma]

    for metric, (label, color, ls, marker, mevery) in METRIC_STYLES.items():
        subset = df_sig[df_sig['metric'] == metric].sort_values('sigma_pert')
        ax.plot(subset['sigma_pert'], subset['detection_rate'],
                label=label, color=color, linestyle=ls, marker=marker,
                markersize=MARKERSIZE, markevery=mevery, linewidth=LINEWIDTH)

    ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.2)


def generate_main_figure(sim_dir, output_path):
    """Generate the main text figure: VISIT 8B + RESOURCE 8B at sigma=0.5."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    plot_single_panel(ax1,
                      os.path.join(sim_dir, 'simulation_v2_VISIT_8b.csv'),
                      sigma=0.5, title=r'VISIT 8B ($\sigma = 0.5$)')
    plot_single_panel(ax2,
                      os.path.join(sim_dir, 'simulation_v2_RESOURCE_8b.csv'),
                      sigma=0.5, title=r'RESOURCE 8B ($\sigma = 0.5$)')

    ax1.set_xlabel(r'$\sigma_{\mathrm{pert}}$', fontsize=11)
    ax2.set_xlabel(r'$\sigma_{\mathrm{pert}}$', fontsize=11)
    ax1.set_ylabel('Detection Rate', fontsize=11)
    ax1.legend(fontsize=9, loc='center right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved: {output_path}')


def generate_appendix_figure(sim_dir, question, model, output_path):
    """Generate an appendix figure: 2x2 grid across 4 sigma values."""
    csv_file = os.path.join(sim_dir, f'simulation_v2_{question}_{model}.csv')
    sigmas = [0.0, 0.25, 0.5, 1.0]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharey=True, sharex=True)

    for ax, sigma in zip(axes.flat, sigmas):
        plot_single_panel(ax, csv_file, sigma, rf'$\sigma = {sigma}$')

    axes[1, 0].set_xlabel(r'$\sigma_{\mathrm{pert}}$', fontsize=11)
    axes[1, 1].set_xlabel(r'$\sigma_{\mathrm{pert}}$', fontsize=11)
    axes[0, 0].set_ylabel('Detection Rate', fontsize=11)
    axes[1, 0].set_ylabel('Detection Rate', fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    fig.suptitle(f'{question} --- {model.upper()}', fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate power curve figures')
    parser.add_argument('--sim-dir', type=str,
                        default='../results/simulation_v2',
                        help='Directory containing simulation CSV files')
    parser.add_argument('--output-dir', type=str,
                        default='../../paper',
                        help='Directory for output PDF files')
    args = parser.parse_args()

    generate_main_figure(
        args.sim_dir,
        os.path.join(args.output_dir, 'power_analysis_combined.pdf'))

    for question, model in [('MANAGE', '8b'), ('MANAGE', '70b'),
                            ('RESOURCE', '8b'), ('RESOURCE', '70b'),
                            ('VISIT', '8b'), ('VISIT', '70b')]:
        generate_appendix_figure(
            args.sim_dir, question, model,
            os.path.join(args.output_dir, f'power_analysis_{question}_{model}.pdf'))
