# Transcript oral — soutenance 15 min

Script parlé slide par slide, en langage simple. Les **[→]** indiquent quand
appuyer sur la flèche droite (apparition d'un élément). Les *(mots-clés)*
sont ceux que le jury attend — à placer, même si on reformule le reste.

**Répartition** : ~30 s + 1 min 30 + 2 min 30 + 1 min + 2 min 30 + 2 min +
1 min 30 + 2 min + 1 min + 30 s ≈ 15 min.

---

## Slide 1 — Titre (30 s)

> Bonjour, on va vous présenter notre projet de reinforcement learning.
> L'idée : prendre un environnement Gymnasium qu'on n'a pas vu en cours —
> Taxi — et comparer les trois algorithmes du programme : Monte Carlo,
> SARSA et Q-Learning. On a fait tourner de vraies expériences, et on va
> vous montrer ce qui marche, ce qui casse, et pourquoi.

---

## Slide 2 — L'environnement (1 min 30)

> Taxi, c'est simple : une grille 5 par 5, un taxi, un passager à aller
> chercher, une destination où le déposer. À chaque pas on perd 1 point —
> donc plus on traîne, plus on perd. Si on charge ou dépose le passager au
> mauvais endroit, moins 10. Si on réussit la course, plus 20.
>
> Ce qui est important pour nous : il y a exactement **500 situations
> possibles** *(états)* et **6 actions**. Du coup on peut stocker la valeur
> de chaque combinaison dans un simple tableau de 500 lignes sur
> 6 colonnes — la fameuse *(Q-table)*. Pas besoin de réseau de neurones.
>
> Deux repères pour lire la suite : un agent qui joue au hasard finit
> vers **−770**. Un agent parfait fait environ **+8**. Toute la question,
> c'est : lequel de nos trois algos remonte de −770 à +8, et à quelle
> vitesse.

---

## Slide 3 — Les trois algorithmes (2 min 30)

> Les trois algos remplissent la même Q-table, et jouent pareil : la
> plupart du temps ils prennent la meilleure action connue, et de temps en
> temps une action au hasard pour explorer — c'est le fameux
> *(epsilon-greedy)*. La SEULE différence entre les trois, c'est la valeur
> qu'ils utilisent pour corriger le tableau — ce qu'on appelle la *(cible)*.
>
> **[→] Monte Carlo.** Lui, il attend la **fin de la partie**. Il note le
> score total réellement obtenu, et corrige toutes les cases qu'il a
> visitées. Avantage : c'est le vrai score, pas une estimation.
> Inconvénient : tout le hasard de la partie est dedans — deux parties
> identiques au début peuvent finir très différemment. C'est ce qu'on
> appelle une cible à **forte *(variance)***. Et il n'apprend rien tant
> que la partie n'est pas finie.
>
> **[→] SARSA.** Lui n'attend pas : à **chaque pas**, il corrige la case
> avec « la récompense immédiate, plus ce que vaut le coup que je vais
> **vraiment** jouer ensuite ». Comme il se base sur ce qu'il fait
> vraiment — y compris ses explorations au hasard — on dit qu'il est
> *(on-policy)* : il apprend la valeur de SA façon de jouer.
>
> **[→] Q-Learning.** Pareil, à chaque pas, mais avec une nuance : « la
> récompense immédiate, plus ce que vaudrait le **meilleur** coup
> suivant ». Même s'il explore n'importe comment, il apprend le jeu
> parfait. C'est *(off-policy)* : ce qu'il apprend ne dépend pas de sa
> façon d'explorer. Retenez cette différence, on va la VOIR dans les
> mesures.

---

## Slide 4 — Protocole (1 min)

> Comment on a testé : 12 000 parties d'entraînement par run, et chaque
> run est répété sur plusieurs *(seeds)* — plusieurs graines aléatoires —
> pour ne pas conclure sur un coup de chance. Ensuite on fait varier les
> réglages **un par un** : la vitesse d'apprentissage alpha, l'importance
> du futur gamma, et la dose d'exploration epsilon.
>
> **[→]** Point important pour lire nos courbes : pendant l'entraînement,
> l'agent fait exprès des coups au hasard. Donc la courbe d'entraînement
> mesure un agent qui explore. Pour juger ce qu'il a VRAIMENT appris, à la
> fin on coupe l'exploration et on mesure la politique pure sur
> 500 parties : c'est notre « évaluation greedy ». Les deux peuvent
> raconter des histoires différentes — vous allez le voir sur epsilon.

---

## Slide 5 — Comparaison (2 min 30)

> Premier résultat, tous au même réglage. Sur les courbes : tout le monde
> part de −770, et tout le monde remonte. Q-Learning, en vert, monte le
> plus vite. Et à l'arrivée…
>
> **[→]** …SARSA et Q-Learning atteignent tous les deux le score optimal :
> **+7,9, 100 % de courses réussies, 13 pas par course**. Monte Carlo
> converge aussi, mais reste en dessous : +3,8.
>
> Deux choses à retenir. Un : Q-Learning est le plus rapide, parce que sa
> cible « meilleur coup suivant » propage l'information sans attendre que
> l'agent joue mieux. Deux : à la fin, SARSA et Q-Learning sont
> **identiques**. C'est normal : Taxi est un environnement
> *(déterministe)* — pas de glissade, pas de surprise. La prudence de
> SARSA ne sert à rien ici. Sur CliffWalking, vu en cours, il y a un
> ravin : là, les deux donnent des chemins différents. Ici non.
> Et Monte Carlo traîne à cause de sa variance : chaque partie bruite
> le tableau.

---

## Slide 6 — Alpha (2 min)

> Alpha, c'est la taille du pas de correction : à quel point chaque
> nouvelle info écrase l'ancienne. Regardez le contraste entre les trois
> panneaux : à gauche Monte Carlo s'effondre dès que alpha monte ; au
> milieu SARSA tient mieux ; à droite Q-Learning, toutes les courbes sont
> confondues.
>
> **[→]** Les chiffres : Monte Carlo passe de +1,7 à **−292** dès
> alpha = 0,3. Logique : sa cible est très bruitée, donc un grand pas fait
> qu'une seule mauvaise partie écrase tout le tableau. SARSA tient jusqu'à
> 0,3 mais casse à 0,7. Et Q-Learning fait +7,9 **partout**.
>
> La phrase à retenir : la sensibilité à alpha, c'est un classement par la
> variance de la cible. Cible bruitée = petit pas obligatoire.

---

## Slide 7 — Gamma (1 min 30)

> Gamma, c'est l'importance du futur : à 1, un point dans 100 pas vaut
> autant qu'un point maintenant ; à 0,9, le futur lointain ne compte
> presque plus.
>
> **[→]** Et là, surprise : gamma agit **en sens opposés** sur Monte Carlo
> et SARSA. Monte Carlo est MEILLEUR à 0,9 — parce qu'écraser le futur
> lointain réduit le bruit de sa cible. À gamma = 1, ses parties de
> 200 pas donnent des scores énormes et tous pareils : impossible de
> savoir quelle action était la bonne. SARSA, c'est l'inverse : à 0,9 il
> devient myope — son horizon utile fait à peu près 10 pas, alors qu'une
> course en demande 13. Loin du but, toutes les actions se valent, il ne
> voit plus le but. Q-Learning, encore une fois, encaisse tout.

---

## Slide 8 — Epsilon (2 min)

> Epsilon, c'est la dose d'exploration : la proportion de coups joués au
> hasard. Et cette slide contient notre piège préféré. Regardez les
> courbes : epsilon = 0,01 — quasi pas d'exploration — donne les courbes
> les PLUS hautes. On dirait le meilleur réglage.
>
> **[→]** Sauf que… en évaluation finale, Monte Carlo avec epsilon = 0,01
> réussit **1,3 %** de ses courses. La courbe était belle uniquement parce
> que l'agent répétait la petite routine qu'il connaissait. Moralité : ne
> jamais juger sur la courbe d'entraînement, toujours évaluer la politique
> finale.
>
> Deux autres choses ici. SARSA et Q-Learning survivent au petit epsilon
> grâce à un mécanisme presque accidentel : la Q-table démarre à zéro,
> alors que les vraies valeurs sont négatives — donc les actions jamais
> essayées ont l'air prometteuses, et l'argmax va les tester tout seul.
> C'est *(l'initialisation optimiste)*. Et à epsilon = 0,3, regardez :
> SARSA descend à −5, Q-Learning reste à +7,9. C'est EXACTEMENT la
> différence on-policy / off-policy de la slide 3 : SARSA compte ses
> propres coups de hasard dans ses valeurs, Q-Learning les ignore. On ne
> l'a pas juste appris en cours — on l'a **mesuré**.

---

## Slide 9 — Synthèse (1 min)

> Ce qu'on retient. Sur les 11 configurations testées, Q-Learning est à
> l'optimal 11 fois sur 11, SARSA 8, Monte Carlo 0.
>
> **[→]** Bien réglés, SARSA et Q-Learning résolvent Taxi parfaitement.
> **[→]** Toutes les faiblesses de Monte Carlo ont UNE seule cause : la
> variance de sa cible, parce qu'il n'a pas de *(bootstrap)*.
> **[→]** On-policy contre off-policy, ce n'est pas que de la théorie :
> ça se mesure, sur epsilon.
> **[→]** Et le match nul SARSA/Q-Learning est une propriété de Taxi, qui
> est déterministe — pas une règle générale.

---

## Slide 10 — Questions (30 s)

> Tout le code, les figures et le rapport détaillé sont sur le repo
> GitHub. Merci de votre attention — on est prêts pour vos questions.

*(Les trois cartes de la slide sont nos réponses préparées aux questions
probables — les révéler [→] seulement si la question tombe.)*

---

## Antisèche dernière minute

| Terme | En une phrase |
|---|---|
| Q-table | Tableau 500×6 : « qu'est-ce que ça rapporte de faire l'action a dans la situation s » |
| epsilon-greedy | Meilleure action connue, sauf ε % du temps : au hasard |
| Cible | La valeur vers laquelle on corrige une case de la Q-table |
| Variance | La cible bouge beaucoup d'une partie à l'autre |
| Bootstrap | Corriger une estimation avec une autre estimation (TD), au lieu d'attendre le vrai résultat (MC) |
| On-policy (SARSA) | Apprend la valeur de SA façon de jouer, exploration incluse |
| Off-policy (Q-Learning) | Apprend le jeu optimal, peu importe comment il explore |
| terminated vs truncated | Vraie fin (course réussie) vs coupure arbitraire à 200 pas — on ne les traite pas pareil dans la mise à jour |
