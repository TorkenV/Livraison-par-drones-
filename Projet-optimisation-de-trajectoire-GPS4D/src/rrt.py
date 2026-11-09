import numpy as np

class Node:
    # Noeud de l'arbre RRT
    def __init__(self, x, y, z):
        self.x      = x
        self.y      = y
        self.z      = z
        self.parent = None  # noeud parent dans l'arbre

    # Vecteur coordonnées de la position actuelle 
    def position(self):
        return np.array([self.x, self.y, self.z])


def distance(node_a, node_b):
    # Distance euclidienne entre deux noeuds
    return np.linalg.norm(node_a.position() - node_b.position())


def nearest_node(tree, random_node):
    # Fonction retournant le noeud le plus proche du noeud actuel
    return min(tree, key=lambda node: distance(node, random_node))


def steer(nearest, random_node, step_size):
    # Avance depuis nearest vers random_node d'un pas step_size

    direction = random_node.position() - nearest.position()
    dist      = np.linalg.norm(direction)

    if dist < 1e-6:
        return nearest

    # Normaliser et avancer d'un pas
    direction = direction / dist
    new_pos   = nearest.position() + direction * min(step_size, dist)

    new_node        = Node(*new_pos)
    new_node.parent = nearest
    return new_node


def is_collision_free(node_a, node_b, obstacles, safety_margin):
    # Vérifie que le segment [node_a, node_b] ne traverse aucun obstacle

    # Échantillonner le segment en points intermédiaires
    num_samples = 20
    for i in range(num_samples + 1):
        t     = i / num_samples
        point = node_a.position() + t * (node_b.position() - node_a.position())

        for obs in obstacles:
            if obs.is_collision(point, safety_margin):
                return False  # collision détectée

    return True  # segment libre


# ═══════════════════════════════════════════════════════════════════
# NOUVEAU : Fonctions Airways (identiques à rrt_star_with_airways)
# ═══════════════════════════════════════════════════════════════════

def is_point_in_any_airway(point, airways):
    """
    Vérifie si un point est à l'intérieur d'au moins une route aérienne
    
    Args:
        point (np.array): position [x, y, z]
        airways (list): liste des routes aériennes
    
    Returns:
        bool: True si le point est dans au moins une route
    """
    if not airways or len(airways) == 0:
        return True  # Pas de routes = free flight partout
    
    for airway in airways:
        if is_point_in_airway(point, airway):
            return True
    
    return False


def is_point_in_airway(point, airway):
    """
    Vérifie si un point est dans un couloir aérien cylindrique
    
    Args:
        point (np.array): position [x, y, z]
        airway: objet Airway avec waypoint_start, waypoint_end, width
    
    Returns:
        bool: True si le point est dans le cylindre
    """
    start = np.array(airway.waypoint_start)
    end = np.array(airway.waypoint_end)
    radius = airway.width / 2.0
    
    # Vecteur de l'axe du cylindre
    axis = end - start
    length = np.linalg.norm(axis)
    
    if length < 1e-6:
        return False
    
    axis_normalized = axis / length
    
    # Vecteur du début du cylindre au point
    vec = point - start
    
    # Projection du point sur l'axe
    projection_length = np.dot(vec, axis_normalized)
    
    # Le point doit être entre le début et la fin du cylindre
    if projection_length < 0 or projection_length > length:
        return False
    
    # Distance perpendiculaire à l'axe
    projection_point = start + projection_length * axis_normalized
    perpendicular_distance = np.linalg.norm(point - projection_point)
    
    # Le point est dans le cylindre si sa distance perpendiculaire <= rayon
    return perpendicular_distance <= radius


def sample_in_airways(airways, x_bounds, y_bounds, z_bounds):
    """
    Échantillonne un point aléatoire dans les routes aériennes
    
    Args:
        airways: liste des routes aériennes
        x_bounds, y_bounds, z_bounds: bornes de l'espace
    
    Returns:
        np.array: point [x, y, z] dans une route
    """
    if not airways or len(airways) == 0:
        # Pas de routes, échantillonner dans tout l'espace
        return np.array([
            np.random.uniform(*x_bounds),
            np.random.uniform(*y_bounds),
            np.random.uniform(*z_bounds)
        ])
    
    # Choisir une route aléatoire
    airway = airways[np.random.randint(0, len(airways))]
    
    start = np.array(airway.waypoint_start)
    end = np.array(airway.waypoint_end)
    radius = airway.width / 2.0
    
    # Point aléatoire le long de l'axe
    t = np.random.uniform(0, 1)
    axis_point = start + t * (end - start)
    
    # Décalage aléatoire perpendiculaire (dans le rayon)
    axis = end - start
    axis_normalized = axis / np.linalg.norm(axis)
    
    # Trouver deux vecteurs perpendiculaires
    if abs(axis_normalized[2]) < 0.9:
        perp1 = np.cross(axis_normalized, [0, 0, 1])
    else:
        perp1 = np.cross(axis_normalized, [1, 0, 0])
    
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(axis_normalized, perp1)
    
    # Point aléatoire dans le cercle perpendiculaire
    r = np.random.uniform(0, radius * 0.8)  # 80% du rayon
    theta = np.random.uniform(0, 2 * np.pi)
    
    offset = r * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
    
    return axis_point + offset


def extract_path(goal_node):
    """
    Remonte l'arbre depuis le noeud goal jusqu'à la racine
    pour extraire le chemin

    Args:
        goal_node (Node): noeud d'arrivée

    Returns:
        list[np.array]: liste de positions [x, y, z]
    """
    path = []
    node = goal_node

    while node is not None:
        path.append(node.position())
        node = node.parent

    path.reverse()  # de start vers goal
    return path


# ═══════════════════════════════════════════════════════════════════
# RRT avec Airways
# ═══════════════════════════════════════════════════════════════════

def rrt(start, goal, obstacles, safety_margin,
        airways=None,  # ← NOUVEAU PARAMÈTRE
        step_size=10.0, max_iter=2000, goal_radius=15.0,
        x_bounds=None, y_bounds=None, z_bounds=None):
    """
    Algorithme RRT avec support des routes aériennes

    Args:
        start         (array-like) : position de départ [x, y, z]
        goal          (array-like) : position d'arrivée [x, y, z]
        obstacles     (list)       : liste d'objets Obstacle
        safety_margin (float)      : marge de sécurité autour des obstacles (m)
        airways       (list)       : NOUVEAU - liste des routes aériennes
        step_size     (float)      : longueur de chaque pas RRT (m)
        max_iter      (int)        : nombre maximum d'itérations
        goal_radius   (float)      : rayon pour considérer le goal atteint (m)
        x_bounds      (tuple)      : (x_min, x_max) espace de recherche
        y_bounds      (tuple)      : (y_min, y_max) espace de recherche
        z_bounds      (tuple)      : (z_min, z_max) espace de recherche

    Returns:
        path (list[np.array]) : liste de points [x, y, z] du start au goal
                                None si aucun chemin trouvé
    """

    start = np.array(start, dtype=float)
    goal  = np.array(goal,  dtype=float)
    
    # Gérer les airways
    if airways is None:
        airways = []
    
    has_airways = len(airways) > 0

    # Bornes par défaut de l'espace de recherche
    if x_bounds is None:
        x_bounds = (min(start[0], goal[0]) - 50, max(start[0], goal[0]) + 50)
    if y_bounds is None:
        y_bounds = (min(start[1], goal[1]) - 200, max(start[1], goal[1]) + 200)
    if z_bounds is None:
        z_bounds = (0, 200)

    # Initialiser l'arbre avec le noeud de départ
    start_node = Node(*start)
    tree       = [start_node]

    print(f"  RRT : {len(airways)} route(s) aérienne(s)")
    print(f"  RRT : départ={start.tolist()}, goal={goal.tolist()}")
    print(f"  RRT : step={step_size}m, max_iter={max_iter}, goal_radius={goal_radius}m")
    
    # Zone libre autour du start/goal (30m de rayon)
    free_zone_radius = 30.0
    
    violations = 0  # Compteur de violations des airways

    for i in range(max_iter):

        # 1. Échantillonner un point aléatoire
        if np.random.rand() < 0.10:
            # 10% du temps : tirer vers le goal
            random_node = Node(*goal)
        else:
            # 90% du temps : échantillonner
            if has_airways and np.random.rand() < 0.95:
                # 95% dans les airways
                random_pos = sample_in_airways(airways, x_bounds, y_bounds, z_bounds)
            else:
                # 5% partout (exploration)
                rx = np.random.uniform(*x_bounds)
                ry = np.random.uniform(*y_bounds)
                rz = np.random.uniform(*z_bounds)
                random_pos = np.array([rx, ry, rz])
            
            random_node = Node(*random_pos)

        # 2. Trouver le noeud le plus proche dans l'arbre
        nearest = nearest_node(tree, random_node)

        # 3. Avancer d'un pas vers le point aléatoire
        new_node = steer(nearest, random_node, step_size)

        # 4. Vérifier collision avec obstacles
        if not is_collision_free(nearest, new_node, obstacles, safety_margin):
            continue

        # 5. NOUVEAU : Vérifier que le noeud est dans les airways
        #    (sauf dans la zone libre autour du start/goal)
        new_pos = new_node.position()
        dist_to_start = np.linalg.norm(new_pos - start)
        dist_to_goal = np.linalg.norm(new_pos - goal)
        
        in_free_zone = (dist_to_start < free_zone_radius) or (dist_to_goal < free_zone_radius)
        
        if has_airways and not in_free_zone:
            if not is_point_in_any_airway(new_pos, airways):
                violations += 1
                continue  # Point hors des airways, on réessaie

        # 6. Ajouter le nouveau noeud à l'arbre
        tree.append(new_node)

        # 7. Vérifier si on est assez proche du goal
        goal_node = Node(*goal)
        if distance(new_node, goal_node) <= goal_radius:
            # Vérifier que le dernier segment vers le goal est libre
            if is_collision_free(new_node, goal_node, obstacles, safety_margin):
                goal_node.parent = new_node
                tree.append(goal_node)

                path = extract_path(goal_node)
                print(f"  RRT : chemin trouvé en {i+1} itérations, {len(path)} points")
                if has_airways:
                    print(f"  RRT : violations airways ignorées : {violations}")
                return path

    # Aucun chemin trouvé
    print(f"  RRT : aucun chemin trouvé après {max_iter} itérations")
    if has_airways:
        print(f"  RRT : violations airways : {violations}")
    return None


# ------------------------------------------------------------------ #
# Test du module                                                        #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from load_map import Map

    print("="*55)
    print("  TEST DU MODULE RRT AVEC AIRWAYS")
    print("="*55)

    for map_id in range(1, 7):
        print(f"\n--- Map {map_id} ---")

        m = Map()
        try:
            m.load_from_xml(f"../data/map{map_id}.xml")
        except:
            print(f"  Map {map_id} non trouvée, ignorée")
            continue

        path = rrt(
            start         = m.start,
            goal          = m.goal,
            obstacles     = m.obstacles,
            airways       = m.airways,  # ← NOUVEAU
            safety_margin = 10.0,
            step_size     = 10.0,
            max_iter      = 2000,
            goal_radius   = 15.0
        )

        if path is not None:
            print(f"  ✅ Chemin : {len(path)} points")
            print(f"     Premier point : {path[0].tolist()}")
            print(f"     Dernier point : {path[-1].tolist()}")
        else:
            print(f"  ❌ Aucun chemin trouvé")

    print("\n" + "="*55)
    print("  Test terminé")
    print("="*55)