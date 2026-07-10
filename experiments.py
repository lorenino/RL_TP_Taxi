"""Banc d'experiences du projet Taxi-v4 : comparaison des 3 algos + hyperparametres.

Deux volets (enonce points 3 et 4) :
  1. Comparaison Monte Carlo / SARSA / Q-Learning a config identique,
     moyennee sur plusieurs seeds (courbes d'apprentissage + evaluation
     greedy finale + baseline aleatoire).
  2. Influence des hyperparametres : on fait varier UN facteur a la fois
     autour de la config de reference (alpha, puis gamma, puis epsilon),
     pour les trois algos.

Usage :
    python experiments.py --quick    # validation rapide (comparaison seule)
    python experiments.py            # run complet (~30 min, tout + figures)

Sorties (dans le dossier du script) :
    comparaison_algos.png, hyperparametre_{alpha,gamma,epsilon}.png,
    resultats.json (toutes les metriques, pour le rapport).
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algos import ALGOS, evaluate

OUTDIR = Path(__file__).resolve().parent

# Config de reference : alpha et gamma classiques pour Taxi, epsilon en
# decroissance lineaire 1.0 -> 0.05 (None = schedule par defaut d'algos.py)
BASE = {"alpha": 0.1, "gamma": 0.99, "epsilon": None}

GRIDS = {
    "alpha": [0.05, 0.1, 0.3, 0.7],
    "gamma": [0.90, 0.99, 1.0],
    "epsilon": [0.01, 0.1, 0.3, None],   # None = decay de reference
}

COLORS = {"Monte Carlo": "tab:red", "SARSA": "tab:blue", "Q-Learning": "tab:green"}
EVAL_EPISODES = 500


def moving_average(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Moyenne glissante alignee sur la FIN de la fenetre (cf. TP2)."""
    window = min(window, max(1, len(x) // 10))
    avg = np.convolve(x, np.ones(window) / window, mode="valid")
    return np.arange(window - 1, len(x)), avg


def run_algo(name: str, n_episodes: int, seeds: list[int], **params) -> dict:
    """Lance un algo sur plusieurs seeds -> courbes empilees + eval greedy."""
    curves, evals = [], []
    for seed in seeds:
        Q, rewards = ALGOS[name](n_episodes=n_episodes, seed=seed, **params)
        curves.append(rewards)
        evals.append(evaluate(Q, EVAL_EPISODES, seed=seed + 10_000))
    return {
        "curves": np.stack(curves),
        "eval": {k: float(np.mean([e[k] for e in evals])) for k in evals[0]},
    }


def fmt(params: dict) -> str:
    eps = params["epsilon"]
    return (f"alpha={params['alpha']}, gamma={params['gamma']}, "
            f"eps={'decay' if eps is None else eps}")


# ---------------------------------------------------------------------------
# Volet 1 : comparaison des trois algos
# ---------------------------------------------------------------------------

def exp_comparaison(n_episodes: int, seeds: list[int], window: int) -> dict:
    print(f"=== Comparaison des 3 algos ({fmt(BASE)}, "
          f"{n_episodes:,} episodes, {len(seeds)} seeds) ===")
    baseline = evaluate(None, EVAL_EPISODES, seed=0)
    print(f"  baseline aleatoire : reward moyen = "
          f"{baseline['reward_moyen']:+.1f}")

    results = {"baseline": {"eval": baseline}}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [2, 1]})
    for name in ALGOS:
        t0 = time.perf_counter()
        res = run_algo(name, n_episodes, seeds, **BASE)
        results[name] = {"eval": res["eval"]}
        print(f"  {name:<12} greedy : reward = {res['eval']['reward_moyen']:+6.1f} "
              f"| succes = {res['eval']['taux_succes']:5.1%} "
              f"| {res['eval']['longueur_moyenne']:5.1f} pas/episode "
              f"({time.perf_counter() - t0:.0f} s)")

        mean_curve = res["curves"].mean(axis=0)
        std_curve = res["curves"].std(axis=0)
        x, avg = moving_average(mean_curve, window)
        _, std = moving_average(std_curve, window)
        ax1.plot(x, avg, color=COLORS[name], linewidth=1.3, label=name)
        ax1.fill_between(x, avg - std, avg + std, color=COLORS[name], alpha=0.15)

        bar = ax2.bar(name, res["eval"]["reward_moyen"], color=COLORS[name])
        ax2.bar_label(bar, fmt="%+.1f", fontsize=9)

    ax1.axhline(7.9, color="gray", linestyle="--", linewidth=0.8,
                label="~ +7.9 : optimal")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(f"Reward par episode (fenetre {window})")
    ax1.set_ylim(bottom=-800)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Courbes d'apprentissage (moyenne +/- ecart-type des seeds)")

    ax2.axhline(baseline["reward_moyen"], color="gray", linestyle=":",
                label=f"aleatoire ({baseline['reward_moyen']:+.0f})")
    ax2.axhline(7.9, color="gray", linestyle="--")
    ax2.set_ylabel("Reward moyen (politique greedy finale)")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_title(f"Evaluation greedy ({EVAL_EPISODES} episodes)")

    fig.suptitle(f"Taxi-v4 — Monte Carlo vs SARSA vs Q-Learning ({fmt(BASE)})")
    fig.tight_layout()
    fig.savefig(OUTDIR / "comparaison_algos.png", dpi=120)
    plt.close(fig)
    return results


# ---------------------------------------------------------------------------
# Volet 2 : influence des hyperparametres (un facteur a la fois)
# ---------------------------------------------------------------------------

def exp_hyperparametre(hp: str, values: list, n_episodes: int,
                       seeds: list[int], window: int) -> dict:
    print(f"=== Influence de {hp} (valeurs {values}, base {fmt(BASE)}) ===")
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for ax, name in zip(axes, ALGOS):
        results[name] = {}
        for i, v in enumerate(values):
            params = {**BASE, hp: v}
            res = run_algo(name, n_episodes, seeds, **params)
            label = "decay" if v is None else str(v)
            results[name][label] = res["eval"]
            print(f"  {name:<12} {hp} = {label:<5} -> greedy "
                  f"{res['eval']['reward_moyen']:+6.1f} "
                  f"(succes {res['eval']['taux_succes']:5.1%})")
            x, avg = moving_average(res["curves"].mean(axis=0), window)
            ax.plot(x, avg, linewidth=1.2, label=f"{hp}={label}",
                    color=plt.cm.viridis(i / max(1, len(values) - 1)))
        ax.set_title(name)
        ax.set_xlabel("Episode")
        ax.set_ylim(bottom=-800)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"Reward par episode (fenetre {window})")

    fig.suptitle(f"Taxi-v4 — influence de {hp} (autres facteurs fixes : base)")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"hyperparametre_{hp}.png", dpi=120)
    plt.close(fig)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiences projet Taxi-v4")
    parser.add_argument("--quick", action="store_true",
                        help="validation rapide : comparaison seule, "
                             "2000 episodes, 1 seed")
    parser.add_argument("--episodes", type=int, default=12_000)
    parser.add_argument("--seeds", type=int, default=5,
                        help="nb de seeds pour la comparaison (grilles : 3)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.quick:
        n_episodes, seeds_cmp, seeds_grid, window = 2_000, [42], [42], 100
    else:
        n_episodes = args.episodes
        seeds_cmp = list(range(42, 42 + args.seeds))
        seeds_grid = list(range(42, 42 + 3))
        window = 200

    all_results = {"config": {"episodes": n_episodes, "base": fmt(BASE)}}
    all_results["comparaison"] = exp_comparaison(n_episodes, seeds_cmp, window)

    if not args.quick:
        for hp, values in GRIDS.items():
            all_results[hp] = exp_hyperparametre(hp, values, n_episodes,
                                                 seeds_grid, window)

    # ponytail: self-check minimal — casse si un algo n'apprend pas.
    # Seuils de convergence calibres pour le run complet (12k episodes) ;
    # en --quick on verifie juste que chacun bat nettement l'aleatoire.
    cmp_ = all_results["comparaison"]
    base_r = cmp_["baseline"]["eval"]["reward_moyen"]
    td_floor, mc_floor = (base_r + 500, base_r + 500) if args.quick else (6, base_r + 200)
    assert cmp_["Q-Learning"]["eval"]["reward_moyen"] > td_floor, \
        f"Q-Learning n'a pas converge : {cmp_['Q-Learning']['eval']}"
    assert cmp_["SARSA"]["eval"]["reward_moyen"] > td_floor, \
        f"SARSA n'a pas converge : {cmp_['SARSA']['eval']}"
    assert cmp_["Monte Carlo"]["eval"]["reward_moyen"] > mc_floor, \
        f"MC ne bat pas nettement l'aleatoire : {cmp_['Monte Carlo']['eval']}"
    print("\nSelf-check OK : les 3 algos apprennent.")

    with open(OUTDIR / "resultats.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Termine en {(time.perf_counter() - t0) / 60:.1f} min. "
          f"Figures + resultats.json dans : {OUTDIR}")


if __name__ == "__main__":
    main()
