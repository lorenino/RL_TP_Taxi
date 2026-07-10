# Projet RL — Monte Carlo vs SARSA vs Q-Learning sur Taxi-v4

> Énoncé : choisir un environnement Gymnasium non traité en cours,
> implémenter **Monte Carlo**, **SARSA** et **Q-Learning**, comparer leurs
> performances à l'aide d'expériences, analyser l'influence des
> hyperparamètres (α, γ, ε). Livrables : code Python, rapport succinct,
> présentation orale (15 min).

Environnement choisi : **`Taxi-v4`** (500 états discrets, 6 actions) —
FrozenLake, CliffWalking et Blackjack ayant été traités en cours/TP.

## L'environnement Taxi-v4

Le taxi doit aller chercher un passager puis le déposer à sa destination,
sur une grille 5×5 avec des murs.

| Élément | Valeur |
|---|---|
| États | 500 = 25 cases taxi × 5 positions passager (4 arrêts + à bord) × 4 destinations |
| Actions | 6 : sud, nord, est, ouest, pickup, dropoff |
| Récompenses | −1 par pas · +20 dropoff réussi · −10 pickup/dropoff illégal |
| Fin d'épisode | dropoff réussi (`terminated`) ou 200 pas (`truncated`, wrapper TimeLimit) |
| Optimal | ≈ +7,9 de reward moyen (trajet moyen ~13 pas) |

Assez petit pour du **Q tabulaire** (`np.zeros((500, 6))`), assez riche pour
que les récompenses trompeuses (−10) et l'horizon long différencient les
algorithmes.

## Les trois algorithmes

Même squelette (Q-table + politique de comportement ε-greedy), seule la
**cible de mise à jour** change :

| Algo | Cible de `Q(s, a)` | Nature |
|---|---|---|
| Monte Carlo (first-visit) | `G` = retour complet observé en fin d'épisode | model-free, sans bootstrap |
| SARSA | `r + γ·Q(s', a')` avec `a'` **réellement jouée** ensuite | TD **on-policy** |
| Q-Learning | `r + γ·max_a' Q(s', a')` | TD **off-policy** |

Ce que ça implique (à dire à l'oral) :

- **MC** n'apprend qu'en fin d'épisode et sa cible a une forte variance
  (tout le hasard de l'épisode entre dans `G`) → convergence la plus lente,
  surtout au début quand les épisodes font 200 pas quasi aléatoires.
- **SARSA** apprend la valeur de la politique *qu'il joue vraiment*,
  exploration comprise → prudent tant que ε > 0.
- **Q-Learning** apprend directement la politique *greedy optimale* même en
  explorant → typiquement le plus rapide sur Taxi.

## Choix d'implémentation assumés

- **MC à pas constant** `Q += α·(G − Q)` (et non la moyenne 1/N) : (1)
  l'énoncé demande l'influence de α sur les *trois* algos ; (2) en contrôle,
  la politique change en permanence → cible non-stationnaire, un pas
  constant oublie les retours obsolètes (Sutton & Barto §6.1).
- **`terminated` vs `truncated`** (piège gymnasium) : `terminated` = vraie
  fin du MDP → cible = `r` seul ; `truncated` = artefact du TimeLimit à
  200 pas → le futur existe encore, on **bootstrappe normalement** sur `s'`.
  Confondre les deux biaise Q près de la limite de temps.
- **Même schedule ε pour les trois** (défaut : décroissance linéaire
  1,0 → 0,05 sur 80 % des épisodes) pour que la comparaison soit à
  exploration égale.
- Tie-break aléatoire de l'argmax (sinon biais vers l'action 0 quand Q = 0).

## Expériences

1. **Comparaison** (`comparaison_algos.png`) : les 3 algos à config
   identique (α = 0,1, γ = 0,99, ε decay), 12 000 épisodes × 5 seeds —
   courbes moyennées ± écart-type + évaluation greedy finale (500 épisodes)
   vs baseline aléatoire.
2. **Hyperparamètres** (`hyperparametre_{alpha,gamma,epsilon}.png`) : un
   facteur varie, les autres restent à la config de référence, 3 seeds :
   - α ∈ {0,05 · 0,1 · 0,3 · 0,7}
   - γ ∈ {0,90 · 0,99 · 1,0}
   - ε ∈ {0,01 · 0,1 · 0,3 · decay}

Toutes les métriques finales sont dans `resultats.json`.

## Lancer

```powershell
# env conda rl-gym : python 3.11, gymnasium 1.3.0, numpy, matplotlib
& "C:\Users\lorenzo\miniconda3\Scripts\conda.exe" run --no-capture-output -n rl-gym python -u experiments.py --quick   # validation ~15 s
& "C:\Users\lorenzo\miniconda3\Scripts\conda.exe" run --no-capture-output -n rl-gym python -u experiments.py           # run complet ~30 min
```

> ⚠️ Passer par `conda run` (ou activer l'env) : en appel direct de
> `python.exe`, les DLL natives (freetype…) manquent et matplotlib crashe
> au `savefig` (erreur Windows `0xC06D007F`).

Le script se termine par un self-check (`assert`) qui casse si un des trois
algos n'a pas appris.

## Résultats

*(run complet en cours — section complétée à l'issue)*

## Fichiers

- `algos.py` — les 3 algorithmes + ε-greedy + évaluation greedy
- `experiments.py` — banc d'expériences, figures, `resultats.json`
- `*.png` — figures pour le rapport / les slides

## Références

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2e éd.) —
  MC control §5.4, SARSA §6.4, Q-Learning §6.5, pas constant §6.1.
- [Documentation Taxi (Gymnasium)](https://gymnasium.farama.org/environments/toy_text/taxi/)
