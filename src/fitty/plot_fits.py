import numpy as np
import matplotlib.pyplot as plt
import fitty.fit_monod as fm

def plot_grid_of_monod_fits(data: list[dict], fit_pars: dict, consistent_axes: bool = False):
    # ------------------------------------------------------------
    # Plot fits for all experiments
    # ------------------------------------------------------------
    figsize = (10,4)
    n_exp = len(data)
    ncols = 3
    nrows = int(n_exp / ncols) + 1

    # find global min/max for consistent axes
    X_min = min([min(exp['X_obs']) for exp in data])
    X_max = max([max(exp['X_obs']) for exp in data])
    t_min = min([min(exp['t']) for exp in data])
    t_max = max([max(exp['t']) for exp in data])
    S_min = min([min(exp['S_obs']) for exp in data if 'S_obs' in exp and exp['S_obs'] is not None])
    S_max = max([max(exp['S_obs']) for exp in data if 'S_obs' in exp and exp['S_obs'] is not None])

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axes = np.atleast_1d(axes).flatten()
    i = 0
    for i, exp in enumerate(data):
        ax = axes[i]
        t = exp['t']
        X_obs = exp['X_obs']
        S0 = exp['S0']
        X0_fit = fit_pars['X0'][i]

        # simulate fit curves
        X_fit, S_fit = fm.integrate_monod(t,
                                        fit_pars['mu_max'], fit_pars['Ks'], fit_pars['Y'],
                                        X0_fit, S0)

        # biomass plot
        ax.scatter(t, X_obs, color='k', s=20, label='X_obs')
        ax.plot(t, X_fit, label='fit X(t)', color='tab:blue')

        # substrate on right axis
        ax2 = ax.twinx()
        ax2.plot(t, S_fit, color='tab:orange', label='fit S(t)')
        if 'S_obs' in exp and exp['S_obs'] is not None:
            ax2.scatter(t, exp['S_obs'], color='tab:red', s=20, label='S_obs')

        if consistent_axes:
            ax.set_xlim(t_min, t_max)
            ax.set_ylim(X_min * 0.9, X_max * 1.1)  # add 10% padding
            if S_min is not None and S_max is not None:
                ax2.set_ylim(S_min * 0.9, S_max * 1.1)


        ax.set_xlabel('time [h]')
        ax.set_ylabel('biomass (X)', color='tab:blue')
        ax2.set_ylabel('substrate (S)', color='tab:orange')
        ax.set_title(f'Strain {exp["strain"]} - {exp["carbon_source"]} - {exp["S0"]} g/L')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # combine legends from both axes
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles1 + handles2, labels1 + labels2, loc='best', fontsize=8)

    # turn off unused subplots (if any)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()