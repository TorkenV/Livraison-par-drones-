"""
Algorithme RRT* classique sans pondération des routes aériennes.
Auteurs: Chadha Raouafi, Basma Farchi, Ahmed Bousbih
Projet: GPS 4D Collaboratif pour VTOL
INSA Toulouse - 2025-2026
"""

import numpy as np


# ------------------------------------------------------------------ #
# Classe noeud                                                        #
# ------------------------------------------------------------------ #

class Node:
    """Noeud de l'arbre RRT* avec coût"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0.0

    def position(self):
        return np.array([self.x, self.y, self.z])


# ------------------------------------------------------------------ #
# Fonctions utilitaires                                               #
# ------------------------------------------------------------------ #

def distance(node_a, node_b):
    """Distance euclidienne entre deux noeuds"""
    return np.linalg.norm(node_a.position() - node_b.position())


def weighted_edge_cost(node_a, node_b, airways, alpha=10.0):
    """
    Coût pondéré entre deux noeuds.

    Si node_b est DANS un airway  → coût = distance euclidienne (normal)
    Si node_b est HORS airway     → coût = distance * alpha (pénalisé)

    Args:
        node_a, node_b : noeuds entre lesquels on calcule le coût
        airways (list) : liste d'objets Airway
        alpha (float)  : facteur de pénalité hors airway (défaut = 10)

    Returns:
        float: coût pondéré de l'arête
    """
    dist = np.linalg.norm(node_a.position() - node_b.position())

    if not airways:
        return dist

    if is_point_in_any_airway(node_b.position(), airways):
        return dist

    return dist * alpha


def nearest_node(tree, random_node):
    """Trouve le noeud le plus proche dans l'arbre"""
    return min(tree, key=lambda node: distance(node, random_node))


def steer(nearest, random_node, step_size, airways=None):
    """
    Avance depuis nearest vers random_node d'un pas step_size.
    Le coût du nouveau noeud intègre la pénalité airways.
    """
    direction = random_node.position() - nearest.position()
    dist = np.linalg.norm(direction)

    if dist < 1e-6:
        return nearest

    direction = direction / dist
    new_pos = nearest.position() + direction * min(step_size, dist)

    new_node = Node(*new_pos)
    new_node.parent = nearest
    new_node.cost = nearest.cost + weighted_edge_cost(nearest, new_node, airways)

    return new_node


def is_collision_free(node_a, node_b, obstacles, safety_margin):
    """Vérifie que le segment [node_a, node_b] ne traverse aucun obstacle"""
    num_samples = 20
    for i in range(num_samples + 1):
        t = i / num_samples
        point = node_a.position() + t * (node_b.position() - node_a.position())

        for obs in obstacles:
            if obs.is_collision(point, safety_margin):
                return False

    return True


def extract_path(goal_node):
    """Remonte l'arbre depuis le goal jusqu'à la racine"""
    path = []
    node = goal_node

    while node is not None:
        path.append(node.position())
        node = node.parent

    path.reverse()
    return path


def find_near_nodes(tree, new_node, search_radius):
    """Trouve tous les noeuds dans un rayon autour de new_node"""
    near_nodes = []
    for node in tree:
        if distance(node, new_node) <= search_radius:
            near_nodes.append(node)
    return near_nodes


def choose_parent(near_nodes, new_node, obstacles, safety_margin, airways=None):
    """
    Choisit le meilleur parent pour new_node parmi near_nodes.
    Utilise le coût pondéré airways pour favoriser les chemins dans les couloirs.
    """
    if not near_nodes:
        return new_node

    min_cost = float('inf')
    best_parent = None

    for candidate in near_nodes:
        if not is_collision_free(candidate, new_node, obstacles, safety_margin):
            continue

        potential_cost = candidate.cost + weighted_edge_cost(
            candidate, new_node, airways
        )

        if potential_cost < min_cost:
            min_cost = potential_cost
            best_parent = candidate

    if best_parent is not None:
        new_node.parent = best_parent
        new_node.cost = min_cost

    return new_node


def rewire(tree, near_nodes, new_node, obstacles, safety_margin, airways=None):
    """
    Réoptimise les noeuds voisins.
    Utilise le coût pondéré airways pour favoriser les chemins dans les couloirs.
    """
    for near_node in near_nodes:
        if near_node == new_node or near_node == new_node.parent:
            continue

        if not is_collision_free(new_node, near_node, obstacles, safety_margin):
            continue

        potential_cost = new_node.cost + weighted_edge_cost(
            new_node, near_node, airways
        )

        if potential_cost < near_node.cost:
            near_node.parent = new_node
            near_node.cost = potential_cost


# ------------------------------------------------------------------ #
# Fonctions spécifiques aux airways                                   #
# ------------------------------------------------------------------ #

def is_point_in_any_airway(point, airways, tolerance=0.0):
    """
    Vérifie si un point est dans AU MOINS UN couloir aérien.

    Args:
        point     (array) : point à tester [x, y, z]
        airways   (list)  : liste d'objets Airway
        tolerance (float) : tolérance supplémentaire (m)

    Returns:
        bool: True si le point est dans au moins un couloir
    """
    if not airways:
        return True

    for airway in airways:
        if airway.is_point_in_corridor(point, tolerance):
            return True

    return False


def is_direction_compliant(node_a, node_b, airways):
    """
    Vérifie que le segment [node_a → node_b] respecte le sens de circulation
    de chaque airway qu'il traverse.

    La vérification repose sur le produit scalaire entre le vecteur de
    déplacement du drone et le vecteur directeur normalisé de l'airway :
        - dot > 0 : angle < 90° → même sens général     ✅
        - dot = 0 : perpendiculaire                      ✅ (toléré)
        - dot < 0 : angle > 90° → sens inverse           ❌ rejeté

    Note : on ne demande pas une colinéarité exacte — le drone peut
    traverser l'airway de biais, du moment qu'il a une composante
    dans le bon sens.

    Args:
        node_a (Node) : noeud de départ du segment
        node_b (Node) : noeud d'arrivée du segment
        airways (list): liste d'objets Airway

    Returns:
        bool: True si le déplacement est conforme au sens de toutes
              les airways traversées, False sinon.
    """
    if not airways:
        return True

    movement = node_b.position() - node_a.position()
    norm = np.linalg.norm(movement)

    # Segment de longueur nulle → pas de direction → on accepte
    if norm < 1e-6:
        return True

    for airway in airways:
        # On vérifie uniquement si node_b est dans le couloir
        if airway.is_point_in_corridor(node_b.position()):
            dot = np.dot(movement, airway.direction_normalized)
            if dot < 0:
                # Le drone entre dans l'airway à contre-sens → rejeté
                return False

    return True


def sample_in_airways(airways, x_bounds, y_bounds, z_bounds):
    """
    Tire un point aléatoire À L'INTÉRIEUR d'un des couloirs.
    Utilisé pour biaiser l'échantillonnage vers les airways (70% du temps).

    Args:
        airways (list): liste d'objets Airway
        x_bounds, y_bounds, z_bounds: bornes de l'espace

    Returns:
        np.array: point aléatoire [x, y, z] dans un couloir
    """
    if not airways:
        return np.array([
            np.random.uniform(*x_bounds),
            np.random.uniform(*y_bounds),
            np.random.uniform(*z_bounds)
        ])

    airway = np.random.choice(airways)

    # Tirer un point le long de l'axe dans le bon sens (start → end)
    t = np.random.rand()
    point_on_axis = airway.waypoint_start + t * airway.direction

    # Décalage radial aléatoire dans le rayon du tube
    radius = airway.width / 2
    r = np.sqrt(np.random.rand()) * radius
    theta = np.random.rand() * 2 * np.pi

    direction = airway.direction_normalized
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, [0, 0, 1])
    else:
        perp1 = np.cross(direction, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    offset = r * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
    point = point_on_axis + offset

    point[2] = np.clip(point[2], airway.altitude_min, airway.altitude_max)

    return point


# ------------------------------------------------------------------ #
# RRT* avec préférence airways (coût pondéré) + sens de circulation  #
# ------------------------------------------------------------------ #

def rrt_star(start, goal, obstacles, airways=None, safety_margin=10.0,
                          step_size=10.0, max_iter=1000, goal_radius=15.0,
                          x_bounds=None, y_bounds=None, z_bounds=None):
    """
    RRT* qui FAVORISE les couloirs aériens via une fonction de coût pondérée
    et RESPECTE leur sens de circulation.

    Le drone n'est PAS obligé de rester dans les airways — il peut en sortir
    mais cela lui coûte plus cher (facteur airway_penalty).
    En revanche, s'il entre dans un airway à contre-sens, le noeud est rejeté.

    Args:
        start, goal       : positions de départ et d'arrivée
        obstacles         : liste d'obstacles
        airways           : liste d'objets Airway
        safety_margin     : marge de sécurité autour des obstacles (m)
        step_size         : longueur du pas RRT (m)
        max_iter          : nombre maximum d'itérations
        goal_radius       : rayon pour considérer le goal atteint (m)
        x_bounds, y_bounds, z_bounds: bornes de l'espace de recherche

    Returns:
        path: liste de points [x,y,z], ou None si échec
    """

    start = np.array(start, dtype=float)
    goal  = np.array(goal,  dtype=float)

    if x_bounds is None:
        x_bounds = (min(start[0], goal[0]) - 50, max(start[0], goal[0]) + 50)
    if y_bounds is None:
        y_bounds = (min(start[1], goal[1]) - 200, max(start[1], goal[1]) + 200)
    if z_bounds is None:
        z_bounds = (0, 200)

    start_node = Node(*start)
    start_node.cost = 0.0
    tree = [start_node]

    gamma = 30.0
    best_goal_node = None
    best_goal_cost = float('inf')


    print(f"  RRT* + Airways : {len(airways)} routes disponibles")
    print(f"  RRT* + Airways : départ={start.tolist()}, goal={goal.tolist()}")
    print(f"  RRT* classique sans pondération airways")
    
    for i in range(max_iter):

        # ── 1. Échantillonnage biaisé ────────────────────────────────
        rand = np.random.rand()

        if rand < 0.70 and airways:
            random_point = sample_in_airways(airways, x_bounds, y_bounds, z_bounds)
            random_node = Node(*random_point)
        elif rand < 0.80:
            random_node = Node(*goal)
        else:
            rx = np.random.uniform(*x_bounds)
            ry = np.random.uniform(*y_bounds)
            rz = np.random.uniform(*z_bounds)
            random_node = Node(rx, ry, rz)

        # ── 2. Noeud le plus proche ──────────────────────────────────
        nearest = nearest_node(tree, random_node)

        # ── 3. Avancer d'un pas ──────────────────────────────────────
        new_node = steer(nearest, random_node, step_size, airways)

        # ── 4. Vérifier collision avec obstacles ─────────────────────
        if not is_collision_free(nearest, new_node, obstacles, safety_margin):
            continue

        # ── 6. Rayon de recherche RRT* ───────────────────────────────
        n = len(tree)
        search_radius = min(gamma * (np.log(n) / n) ** (1.0 / 3.0), step_size * 2)

        # ── 7. Trouver les voisins ───────────────────────────────────
        near_nodes_list = find_near_nodes(tree, new_node, search_radius)

        # ── 8. Choisir le meilleur parent ────────────────────────────
        new_node = choose_parent(
            near_nodes_list, new_node, obstacles, safety_margin, airways
        )

        # ── 9. Ajouter à l'arbre ─────────────────────────────────────
        tree.append(new_node)

        # ── 10. Rewiring ─────────────────────────────────────────────
        rewire(tree, near_nodes_list, new_node, obstacles, safety_margin, airways)

        # ── 11. Vérifier si le goal est atteint ──────────────────────
        goal_node = Node(*goal)
        if distance(new_node, goal_node) <= goal_radius:
            if is_collision_free(new_node, goal_node, obstacles, safety_margin):
                # Vérifier aussi le sens pour le dernier segment vers le goal
                if is_direction_compliant(new_node, goal_node, airways):
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + weighted_edge_cost(
                        new_node, goal_node, airways
                    )

                    if goal_node.cost < best_goal_cost:
                        best_goal_node = goal_node
                        best_goal_cost = goal_node.cost
                        print(f"  RRT* + Airways : chemin trouvé "
                              f"(coût={best_goal_cost:.2f}) iter {i+1}")

    if best_goal_node is not None:
        path = extract_path(best_goal_node)
        print(f"  RRT* + Airways : succès, {len(path)} points, "
              f"coût={best_goal_cost:.2f}")
        
        if airways:
            in_airway = sum(
                1 for p in path if is_point_in_any_airway(p, airways)
            )
            print(f"  RRT* + Airways : {in_airway}/{len(path)} points "
                  f"dans les couloirs ({100*in_airway/len(path):.1f}%)")

        return path
    else:
        print(f"  RRT* + Airways : échec après {max_iter} itérations")
        return None


# ------------------------------------------------------------------ #
# Test                                                                 #
# ------------------------------------------------------------------ #

# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, '.')

#     from load_map import Map

#     print("="*55)
#     print("  TEST RRT* AVEC AIRWAYS (coût pondéré + sens)")
#     print("="*55)

#     m = Map()
#     m.load_from_xml("../data/map5.xml")

#     path = rrt_star_with_airways(
#         start         = m.start,
#         goal          = m.goal,
#         obstacles     = m.obstacles,
#         airways       = m.airways,
#         safety_margin = 10.0,
#         step_size     = 10.0,
#         max_iter      = 500,
#         goal_radius   = 15.0,
#         airway_penalty= 10.0
#     )

#     if path:
#         print(f"\n✅ Chemin trouvé : {len(path)} waypoints")
#     else:
#         print("\n❌ Aucun chemin trouvé")