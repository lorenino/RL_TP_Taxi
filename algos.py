"""Taxi-v4 : les trois algorithmes tabulaires du projet final.

Meme squelette pour les trois : Q-table (500 etats x 6 actions), politique
de comportement epsilon-greedy, et une regle de mise a jour differente :

  - Monte Carlo (first-visit, pas constant alpha) : attend la FIN de
    l'episode, puis met a jour avec le retour complet G reellement observe.
    Cible sans biais de bootstrap, mais variance elevee et aucune mise a
    jour en cours d'episode.
  - SARSA (on-policy TD) : met a jour a CHAQUE pas avec la cible
    r + gamma * Q(s', a') ou a' est l'action que la politique de
    comportement va REELLEMENT jouer -> apprend la valeur de la politique
    epsilon-greedy elle-meme (prudente tant qu'il y a de l'exploration).
  - Q-Learning (off-policy TD) : cible r + gamma * max_a' Q(s', a') ->
    apprend directement la politique optimale, quelle que soit la
    politique de comportement.

Choix assume : MC utilise un pas constant alpha (Q += alpha * (G - Q)) et
non la moyenne 1/N du TP2 Blackjack. Deux raisons : (1) l'enonce demande
d'analyser l'influence de alpha sur LES TROIS algos, il faut donc que les
trois l'aient ; (2) en controle, la politique change en permanence, donc la
cible est non-stationnaire — un pas constant oublie les vieux retours
obsoletes (Sutton & Barto, section 6.1).

Piege gymnasium assume (ca tombe a l'oral) : `terminated` est une fin REELLE
du MDP (dropoff reussi, plus de futur -> cible = r seul), `truncated` est un
artefact du wrapper TimeLimit (200 pas sur Taxi) : le futur existe encore,
donc on bootstrappe normalement sur s'. Confondre les deux biaise Q pres de
la limite de temps. Pour MC, l'episode tronque donne un retour incomplet :
biais connu de la methode, on le mentionne dans le rapport.

Reference : Sutton & Barto, "Reinforcement Learning: An Introduction",
2e ed. — MC control section 5.4, SARSA section 6.4, Q-Learning section 6.5.
"""

import gymnasium as gym
import numpy as np

ENV_ID = "Taxi-v4"
N_STATES = 500   # 25 cases taxi x 5 positions passager x 4 destinations
N_ACTIONS = 6    # sud, nord, est, ouest, pickup, dropoff

EPS_START, EPS_MIN, EPS_DECAY_FRAC = 1.0, 0.05, 0.8


def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float,
                   rng: np.random.Generator) -> int:
    """Action aleatoire avec proba epsilon, sinon argmax (tie-break aleatoire)."""
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    q = Q[state]
    best = np.flatnonzero(q == q.max())
    return int(rng.choice(best))


def epsilon_schedule(k: int, n_episodes: int, epsilon: float | None) -> float:
    """epsilon fixe si fourni, sinon decroissance lineaire 1.0 -> 0.05 sur 80 %.

    Le schedule par defaut (None) sert d'hyperparametre de reference ; les
    valeurs fixes servent a l'experience "influence d'epsilon".
    """
    if epsilon is not None:
        return epsilon
    frac = min(1.0, k / (EPS_DECAY_FRAC * n_episodes))
    return EPS_START - frac * (EPS_START - EPS_MIN)


def _make_env(seed: int) -> tuple[gym.Env, np.random.Generator]:
    env = gym.make(ENV_ID)
    env.reset(seed=seed)   # seed le PRNG de l'env une fois pour toutes
    return env, np.random.default_rng(seed)


def mc_control(n_episodes: int, alpha: float, gamma: float,
               epsilon: float | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """On-policy first-visit Monte Carlo control, pas constant alpha."""
    env, rng = _make_env(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    rewards = np.empty(n_episodes)

    for k in range(n_episodes):
        eps = epsilon_schedule(k, n_episodes, epsilon)

        # 1) generer un episode complet avec la politique courante
        episode = []
        state, _ = env.reset()
        done = False
        while not done:
            action = epsilon_greedy(Q, state, eps, rng)
            next_state, r, terminated, truncated, _ = env.step(action)
            episode.append((state, action, float(r)))
            state = next_state
            done = terminated or truncated

        # 2) indice de premiere visite de chaque paire (s, a)
        first_visit = {}
        for t, (s, a, _) in enumerate(episode):
            first_visit.setdefault((s, a), t)

        # 3) parcours arriere : G accumule les recompenses depuis la fin
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            if first_visit[(s, a)] == t:
                Q[s, a] += alpha * (G - Q[s, a])

        rewards[k] = sum(r for _, _, r in episode)

    env.close()
    return Q, rewards


def sarsa(n_episodes: int, alpha: float, gamma: float,
          epsilon: float | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """SARSA : TD on-policy, cible r + gamma * Q(s', a') avec a' vraiment jouee."""
    env, rng = _make_env(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    rewards = np.empty(n_episodes)

    for k in range(n_episodes):
        eps = epsilon_schedule(k, n_episodes, epsilon)
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, eps, rng)
        total = 0.0
        while True:
            next_state, r, terminated, truncated, _ = env.step(action)
            total += r
            # a' est choisie MAINTENANT et sera jouee au pas suivant :
            # c'est ce qui rend SARSA on-policy
            next_action = epsilon_greedy(Q, next_state, eps, rng)
            target = r if terminated else r + gamma * Q[next_state, next_action]
            Q[state, action] += alpha * (target - Q[state, action])
            if terminated or truncated:
                break
            state, action = next_state, next_action

        rewards[k] = total

    env.close()
    return Q, rewards


def q_learning(n_episodes: int, alpha: float, gamma: float,
               epsilon: float | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Q-Learning : TD off-policy, cible r + gamma * max_a' Q(s', a')."""
    env, rng = _make_env(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    rewards = np.empty(n_episodes)

    for k in range(n_episodes):
        eps = epsilon_schedule(k, n_episodes, epsilon)
        state, _ = env.reset()
        total = 0.0
        while True:
            action = epsilon_greedy(Q, state, eps, rng)
            next_state, r, terminated, truncated, _ = env.step(action)
            total += r
            # max sur a' : on apprend la politique GREEDY meme en explorant
            target = r if terminated else r + gamma * Q[next_state].max()
            Q[state, action] += alpha * (target - Q[state, action])
            if terminated or truncated:
                break
            state = next_state

        rewards[k] = total

    env.close()
    return Q, rewards


def evaluate(Q: np.ndarray | None, n_episodes: int, seed: int) -> dict:
    """Evalue une politique SANS apprendre. Q=None -> baseline aleatoire.

    Politique greedy pure (epsilon = 0) sinon. Renvoie reward moyen, taux
    d'episodes reussis (dropoff avant le timeout) et longueur moyenne.
    """
    env, rng = _make_env(seed)
    epsilon = 1.0 if Q is None else 0.0
    Q = Q if Q is not None else np.zeros((N_STATES, N_ACTIONS))

    total_reward, n_success, total_steps = 0.0, 0, 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        while True:
            action = epsilon_greedy(Q, state, epsilon, rng)
            state, r, terminated, truncated, _ = env.step(action)
            total_reward += r
            total_steps += 1
            if terminated:
                n_success += 1   # sur Taxi, terminated = dropoff reussi
                break
            if truncated:
                break

    env.close()
    return {
        "reward_moyen": total_reward / n_episodes,
        "taux_succes": n_success / n_episodes,
        "longueur_moyenne": total_steps / n_episodes,
    }


ALGOS = {
    "Monte Carlo": mc_control,
    "SARSA": sarsa,
    "Q-Learning": q_learning,
}
