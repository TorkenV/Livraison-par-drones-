# GPS4D — Système de Navigation 4D pour Drones VTOL

> **Projet PIR — INSA Toulouse, 4ème année Mathématiques Appliquées — 2025-2026**  
> Auteurs : Chadha Raouafi, Basma Farchi, Ahmed Bousbih

---

## Présentation

GPS4D est un système de navigation **4D (x, y, z, t)** pour drones de type **VTOL** (plateforme *Private Bee*). L'objectif est de planifier des trajectoires optimales en 3D, d'éviter les obstacles, de respecter les routes aériennes (airways) et de générer un vecteur d'état complet pour chaque point de la trajectoire.

Le scénario de référence est une **livraison de pizza** entre une pizzeria et une résidence étudiante près de l'INSA Toulouse, avec 9 maps de complexité croissante.

---

## Architecture du projet

```
Projet-optimisation-de-trajectoire-GPS4D/
│
├── data/
│   ├── drone1.xml              
│   ├── map1.xml                
│   ├── map2.xml                
│   ├── map3.xml
│   ├── map4.xml
│   ├── map5.xml
│   ├── map6.xml                
│   ├── map7.xml
│   ├── map8.xml
│   └── map9.xml
│
├── results/                    
│
└── src/
    ├── main.py                         ← Pipeline Method 0 : RRT* Airways + L-BFGS Airways + B-Splines
    ├── main1.py                        ← Pipeline Method 1 : RRT + B-Splines (baseline)
    ├── main2.py                        ← Pipeline Method 2 : RRT + L-BFGS + B-Splines
    ├── main3.py                        ← Pipeline Method 3 : RRT* + L-BFGS + B-Splines
    ├── load_drone.py                   ← Classe Drone (chargement XML)
    ├── load_map.py                     ← Classes Map, Obstacle, Airway (chargement XML)
    ├── rrt.py                          ← Algorithme RRT avec support airways
    ├── rrt_star.py                     ← Algorithme RRT* (optimisation de coût)
    ├── rrt_star_with_airways.py        ← RRT* avec préférence et sens airways
    ├── lbfgs_optimizer.py              ← L-BFGS avec pénalité airways
    ├── lbfgs_optimizer2.py             ← L-BFGS sans airways (obstacles seuls)
    ├── splines.py                      ← B-Splines cubiques + vecteur d'état 10D
    ├── visualization.py                ← Visualisation 3D, profils, animation
    └── test_trajet_rectiligne.py       ← Test trajectoire rectiligne (sans RRT)
```

---

## Dépendances

```bash
pip install numpy scipy matplotlib
```

Python 3.8+ requis.

---

## Lancement

Depuis le dossier `src/`, choisir la méthode souhaitée et passer l'identifiant de la map (1 à 9) en argument :

```bash
# Method 0 : RRT* Airways + L-BFGS Airways + B-Splines (pipeline complet)
python main.py <map_id>

# Method 1 : RRT + B-Splines (baseline, sans optimisation)
python main1.py <map_id>

# Method 2 : RRT + L-BFGS (sans airways) + B-Splines
python main2.py <map_id>

# Method 3 : RRT* + L-BFGS (sans airways) + B-Splines
python main3.py <map_id>
```

**Exemple :**
```bash
python main.py 3
```

---

## Description des méthodes

### Method 0 — Pipeline complet avec Airways (`main.py`)

| Étape | Module | Description |
|-------|--------|-------------|
| 1 | `load_drone.py` | Chargement des caractéristiques du drone depuis XML |
| 2 | `load_map.py` | Chargement de la map : obstacles, airways, départ/arrivée |
| 3 | `rrt_star_with_airways.py` | Planification RRT* avec préférence pour les airways et respect du sens de circulation |
| 4 | `splines.py` | Premier lissage B-Splines (50 points) |
| 5 | `lbfgs_optimizer.py` | Optimisation L-BFGS des positions (avec pénalité airways) |
| 6 | `splines.py` | Trajectoire finale + vecteur d'état 10D (200 points) |

### Method 1 — Baseline RRT + B-Splines (`main1.py`)

Pipeline minimal : RRT → B-Splines. Pas d'optimisation L-BFGS. Utile pour comparer la qualité de la trajectoire brute.

### Method 2 — RRT + L-BFGS + B-Splines (`main2.py`)

Identique à Method 1 mais avec une passe d'optimisation L-BFGS (sans prise en compte des airways).

### Method 3 — RRT* + L-BFGS + B-Splines (`main3.py`)

Utilise RRT* (meilleur coût garanti asymptotiquement) à la place de RRT, suivi d'un L-BFGS sans airways.

---

## Modules

### `load_drone.py` — Classe `Drone`

Charge les spécifications du drone depuis un fichier XML (`drone1.xml`). Dans notre cas, on utilise un cas simple de drone avec une vitesse et une accélaration constante et on ne prend pas la cosommation de carburant.

**Attributs principaux :**

| Attribut | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `cruise_speed` | Vitesse de croisière (m/s) | 10.0 |
| `max_speed` | Vitesse maximale (m/s) | 15.0 |
| `max_acceleration` | Accélération max (m/s²) | 2.0 |
| `max_deceleration` | Décélération max (m/s²) | 2.5 |
| `mass_empty` | Masse à vide (kg) | 5.0 |
| `mass_loaded` | Masse en charge (kg) | 6.0 |
| `altitude_min` | Altitude minimale (m) | 20.0 |
| `altitude_max` | Altitude maximale (m) | 120.0 |

---

### `load_map.py` — Classes `Map`, `Obstacle`, `Airway`

Charge la carte depuis un fichier XML. Gère deux formats d'obstacles (center+dimensions ou min/max bounds).

**Classe `Airway` :** Couloir aérien cylindrique avec sens de circulation obligatoire.  
**Classe `Obstacle` :** Boîte englobante alignée sur les axes (AABB).

---

### `rrt.py` — Algorithme RRT

RRT standard avec support des routes aériennes. 

```
rrt(start, goal, obstacles, safety_margin, airways, step_size, max_iter, goal_radius)
```

---

### `rrt_star_with_airways.py` — Algorithme RRT*

RRT* avec :
- **Fonction de coût pondérée** : les noeuds hors airway coûtent `alpha` fois plus cher
- **Sens de circulation** : les segments contre-sens sont rejetés (vérification sur tout le segment, pas seulement le noeud final)
- **Rewiring** : réoptimisation locale des noeuds voisins

```
rrt_star_with_airways(start, goal, obstacles, airways, safety_margin,
                      step_size, max_iter, goal_radius, airway_penalty)
```

---

### `lbfgs_optimizer.py` — Optimisation L-BFGS avec Airways

Optimise les **positions** des waypoints intermédiaires (départ et arrivée fixes).

**Fonction de coût :**
```
coût = longueur + 5 × courbure + pénalité_obstacles + pénalité_airways + pénalité_direction
```

- `pénalité_obstacles` : 1000 par waypoint en collision (hard)
- `pénalité_airways` : `airway_penalty` par waypoint hors couloir (soft, défaut 200)
- `pénalité_direction` : barrière quadratique pour les passages à contre-sens

---

### `lbfgs_optimizer2.py` — Optimisation L-BFGS sans Airways

Version simplifiée sans prise en compte des routes aériennes :
```
coût = longueur + 5 × courbure + pénalité_obstacles
```

---

### `splines.py` — B-Splines cubiques + Vecteur d'état 10D

Lisse les waypoints avec une B-Spline cubique (C²) et calcule le vecteur d'état à chaque instant.

**Vecteur d'état 10D :**

| Indices | Contenu |
|---------|---------|
| `[0:3]` | Position (x, y, z) en mètres |
| `[3:6]` | Vitesse (vx, vy, vz) en m/s |
| `[6:9]` | Accélération (ax, ay, az) en m/s² |
| `[9]`   | Masse (kg) — constante |

La paramétrisation temporelle est faite par longueur d'arc pour maintenir une vitesse approximativement constante égale à `cruise_speed`.

**Fonctions principales :**
- `smooth_path(waypoints, num_points)` → trajectoire lissée
- `build_trajectory(waypoints, mass, cruise_speed, num_points)` → `(states, times, smooth)`

---

### `visualization.py` — Visualisation

Trois modes de visualisation :

| Fonction | Description |
|----------|-------------|
| `plot_trajectory(...)` | Vue 3D statique avec vecteurs vitesse |
| `plot_profiles(...)` | Profils position / vitesse / accélération en fonction du temps |
| `animate_trajectory(...)` | Animation 3D du drone le long de la trajectoire |

Les obstacles sont affichés en bleu (boîtes 3D), les airways en violet (cylindres avec flèche de sens).

---

## Format des fichiers XML

### `drone1.xml`
```xml
<drone>
  <name>Drone alpha</name>
  <cruise_speed>10.0</cruise_speed>
  <max_speed>15.0</max_speed>
  <max_acceleration>2.0</max_acceleration>
  <max_deceleration>2.5</max_deceleration>
  <mass_empty>5.0</mass_empty>
  <mass_loaded>6.0</mass_loaded>
  <altitude_min>20.0</altitude_min>
  <altitude_max>120.0</altitude_max>
</drone>
```

### `mapN.xml`
```xml
<map>
  <n>Nom de la map</n>
  <scenario_id>1</scenario_id>

  <mission>
    <start><x>0</x><y>0</y><z>25</z></start>
    <goal><x>500</x><y>0</y><z>25</z></goal>
  </mission>

  <obstacles>
    <obstacle>
      <n>Bâtiment A</n>
      <center><x>200</x><y>0</y><z>50</z></center>
      <dimensions><width>40</width><depth>40</depth><height>100</height></dimensions>
    </obstacle>
    <!-- Alternative : min_bound / max_bound -->
    <obstacle>
      <n>Bâtiment B</n>
      <min_bound><x>100</x><y>-20</y><z>0</z></min_bound>
      <max_bound><x>140</x><y>20</y><z>80</z></max_bound>
    </obstacle>
  </obstacles>

  <airways>
    <airway>
      <n>Route Nord</n>
      <type>mandatory</type>
      <waypoint_start><x>0</x><y>50</y><z>40</z></waypoint_start>
      <waypoint_end><x>500</x><y>50</y><z>40</z></waypoint_end>
      <width>30</width>
      <altitude_min>20</altitude_min>
      <altitude_max>120</altitude_max>
    </airway>
  </airways>
</map>
```

---

## Repère spatial

Le projet utilise le repère **ENU (East-North-Up)** :

- **X** → Est (m)
- **Y** → Nord (m)
- **Z** → Haut / Altitude (m)

---

## Résumé des sorties console

À chaque simulation, le pipeline affiche un résumé complet :

```
=======================================================
  RÉSUMÉ
=======================================================
  Map              : Scénario 3 - Obstacles multiples
  Drone            : Drone alpha
  Obstacles        : 3
  Routes aériennes : 2
  Waypoints RRT    : 47
  Après L-BFGS     : 50 points
  Points finaux    : 200
  Optimisation     : L-BFGS (-12.4% coût)
  Durée de vol     : 68.32 s
  Vitesse moy      : 9.87 m/s
  Masse            : 6.0 kg (constante)
=======================================================
```

---

## Notes techniques

- Les seeds aléatoires (`np.random.seed(42)`, `random.seed(42)`) sont fixées pour la reproductibilité.
- L-BFGS opère avec des bornes ±50 m en x/y et ±30 m en z autour de chaque waypoint intermédiaire.
- La marge de sécurité autour des obstacles est égale à `drone.max_deceleration` (2.5 m par défaut).
- Le RRT* effectue un rewiring dans un rayon `min(γ × (ln(n)/n)^(1/3), 2 × step_size)`.
- Les airways sont modélisées comme des **cylindres** (pas des prismes rectangulaires) : vérification par distance perpendiculaire à l'axe.
