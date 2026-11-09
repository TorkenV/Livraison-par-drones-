"""
Optimisation de trajectoire avec L-BFGS
Auteurs: Chadha Raouafi, Basma Farchi, Ahmed Bousbih
Projet: GPS 4D Collaboratif pour VTOL
INSA Toulouse - 2025-2026

Pipeline : RRT → B-Splines → L-BFGS
L-BFGS optimise les POSITIONS des waypoints intermédiaires
pour minimiser longueur + courbure + pénalité obstacles
Le départ et l'arrivée restent fixes.
Sans prise en compte des routes aériennes.
"""

import numpy as np
from scipy.optimize import minimize


# ------------------------------------------------------------------ #
# Fonction utilitaire airways                                          #
# ------------------------------------------------------------------ #

def is_point_in_any_airway(point, airways):
    """
    Vérifie si un point est dans au moins un couloir aérien.

    Args:
        point   (np.array) : position [x, y, z]
        airways (list)     : liste d'objets Airway

    Returns:
        bool: True si le point est dans au moins un couloir
    """
    if not airways:
        return True  # pas d'airways = pas de contrainte

    for airway in airways:
        if airway.is_point_in_corridor(point):
            return True

    return False


# ------------------------------------------------------------------ #
# Fonction principale                                                  #
# ------------------------------------------------------------------ #

def optimize_trajectory_lbfgs2(waypoints_rrt, drone, obstacles=None,
                                verbose=True):
    """
    Optimise les positions des waypoints intermédiaires avec L-BFGS.

    Reçoit en entrée les points issus de la B-Spline (déjà lissés une
    première fois) et les repositionne pour minimiser :
        - longueur totale
        - courbure (zigzags)
        - pénalité obstacles (hard : 1000 par collision)
    Pas de prise en compte des routes aériennes.

    Args:
        waypoints_rrt  (list or np.array) : waypoints [[x,y,z], ...]
        drone                             : objet Drone
        obstacles      (list)             : liste d'objets Obstacle
        verbose        (bool)             : afficher les logs

    Returns:
        waypoints_opt  (np.array) : waypoints optimisés, shape (N, 3)
        cost_reduction (float)    : réduction de coût en %
    """

    if verbose:
        print("\n" + "="*55)
        print("  OPTIMISATION L-BFGS (positions)")
        print("="*55)

    waypoints = np.array(waypoints_rrt, dtype=float)
    n = len(waypoints)


    # Cas trivial
    if n <= 2:
        if verbose:
            print("  Moins de 3 waypoints, rien à optimiser.")
        return waypoints, 0.0

    # Départ et arrivée fixes
    start = waypoints[0].copy()
    goal  = waypoints[-1].copy()
    intermediaires = waypoints[1:-1].copy()
    x0 = intermediaires.flatten()

    # ═══════════════════════════════════════════════════════════
    # FONCTION DE COÛT
    # ═══════════════════════════════════════════════════════════

    def cost_function(x):
        """
        Coût total = longueur + courbure + pénalité obstacles

        Obstacles : hard constraint (1000) → jamais violé
        """
        pts = np.vstack([start, x.reshape(-1, 3), goal])



        # ── Longueur totale ──────────────────────────────────────────
        longueur = 0.0
        for i in range(len(pts) - 1):
            longueur += np.linalg.norm(pts[i+1] - pts[i])

        # ── Courbure ─────────────────────────────────────────────────
        courbure = 0.0
        for i in range(1, len(pts) - 1):
            v1 = pts[i]   - pts[i-1]
            v2 = pts[i+1] - pts[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                cos_angle = np.dot(v1 / n1, v2 / n2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                courbure += (1.0 - cos_angle)

        # ── Pénalité obstacles (hard) ─────────────────────────────────
        penalty_obs = 0.0
        if obstacles:
            safety = drone.max_deceleration if hasattr(drone, 'max_deceleration') else 5.0
            for pt in pts[1:-1]:
                for obs in obstacles:
                    if obs.is_collision(pt, safety):
                        penalty_obs += 1000.0




        # ── Coût total ────────────────────────────────────────────────
        cost = longueur + 5.0 * courbure + penalty_obs
        return cost

    # ═══════════════════════════════════════════════════════════
    # BORNES
    # ═══════════════════════════════════════════════════════════

    bounds = []
    for pt in intermediaires:
        bounds.append((pt[0] - 50, pt[0] + 50))
        bounds.append((pt[1] - 50, pt[1] + 50))
        bounds.append((max(pt[2] - 30, 20), pt[2] + 30))

    # ═══════════════════════════════════════════════════════════
    # OPTIMISATION L-BFGS-B
    # ═══════════════════════════════════════════════════════════

    cost_init = cost_function(x0)

    if verbose:
        print(f"  Waypoints intermédiaires : {n-2} (sur {n} au total)")
        print(f"  Variables                : {len(x0)}")
        print(f"  Coût initial             : {cost_init:.2f}")
        print(f"  Optimisation en cours...")

    result = minimize(
        fun=cost_function,
        x0=x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 300,
            'ftol'   : 1e-7,
            'gtol'   : 1e-5,
            'disp'   : False
        }
    )

    cost_final = result.fun
    cost_reduction = ((cost_init - cost_final) / cost_init) * 100 if cost_init > 0 else 0.0

    waypoints_opt = np.vstack([start, result.x.reshape(-1, 3), goal])

    if verbose:
        print(f"  Coût final               : {cost_final:.2f}")
        print(f"  Réduction                : {cost_reduction:.1f}%")
        print(f"  Succès L-BFGS            : {result.success}")
        print(f"  Message                  : {result.message}")

        print("="*55)

    return waypoints_opt, cost_reduction


# ------------------------------------------------------------------ #
# Test du module                                                        #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from load_drone import Drone

    print("="*55)
    print("  TEST DU MODULE L-BFGS2 (positions, sans airways)")
    print("="*55)

    drone = Drone()
    drone.load_from_xml("../data/drone1.xml")

    waypoints_test = [
        [0,   0,  25],
        [100, 80,  30],
        [200, 10,  35],
        [300, 90,  40],
        [400, 20,  35],
        [500,  0,  25]
    ]

    print(f"\n  Waypoints initiaux (avec zigzags) :")
    for i, wp in enumerate(waypoints_test):
        print(f"    {i}: {wp}")

    waypoints_opt, reduction = optimize_trajectory_lbfgs2(
        waypoints_rrt  = waypoints_test,
        drone          = drone,
        verbose        = True
    )

    print(f"\n  Waypoints optimisés :")
    for i, wp in enumerate(waypoints_opt):
        print(f"    {i}: {wp.round(2).tolist()}")
    print(f"  Réduction de coût : {reduction:.1f}%")

    print("\n" + "="*55)
    print("  Test terminé")
    print("="*55)