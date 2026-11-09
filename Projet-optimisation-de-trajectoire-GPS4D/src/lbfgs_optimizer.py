"""
Optimisation de trajectoire avec L-BFGS
Auteurs: Chadha Raouafi, Basma Farchi, Ahmed Bousbih
Projet: GPS 4D Collaboratif pour VTOL
INSA Toulouse - 2025-2026

Pipeline : RRT → B-Splines → L-BFGS
L-BFGS optimise les POSITIONS des waypoints intermédiaires
pour minimiser longueur + courbure + pénalité airways
Le départ et l'arrivée restent fixes.
Le drone FAVORISE les airways mais n'est PAS obligé d'y rester.
"""

import numpy as np
from scipy.optimize import minimize


# ------------------------------------------------------------------ #
# Fonction utilitaire airways                                          #
# ------------------------------------------------------------------ #

def is_point_in_any_airway(point, airways):
    """
    Vérifie si un point est dans au moins un couloir aérien.
    """
    if not airways:
        return True

    for airway in airways:
        if airway.is_point_in_corridor(point):
            return True

    return False


# ------------------------------------------------------------------ #
# Fonction principale                                                  #
# ------------------------------------------------------------------ #

def optimize_trajectory_lbfgs(waypoints_rrt, drone, obstacles=None,
                               airways=None, airway_penalty=200.0,
                               verbose=True):
    """
    Optimise les positions des waypoints intermédiaires avec L-BFGS.

    Minimise :
        - longueur totale
        - courbure (zigzags)
        - pénalité obstacles (hard : 1000 par collision)
        - pénalité airways   (soft : airway_penalty par waypoint hors couloir)
        - pénalité sens de circulation (barrière quadratique)
    """

    if verbose:
        print("\n" + "="*55)
        print("  OPTIMISATION L-BFGS (positions)")
        print("="*55)

    waypoints = np.array(waypoints_rrt, dtype=float)
    n = len(waypoints)

    has_airways = airways is not None and len(airways) > 0

    if verbose and has_airways:
        print(f"  Airways détectées        : {len(airways)}")
        print(f"  Pénalité hors airway     : {airway_penalty} par waypoint")

    if n <= 2:
        if verbose:
            print("  Moins de 3 waypoints, rien à optimiser.")
        return waypoints, 0.0

    start = waypoints[0].copy()
    goal  = waypoints[-1].copy()
    intermediaires = waypoints[1:-1].copy()
    x0 = intermediaires.flatten()

    # ═══════════════════════════════════════════════════════════
    # FONCTION DE COÛT
    # ═══════════════════════════════════════════════════════════

    def cost_function(x):
        """
        Coût total = longueur + courbure + pénalité obstacles + pénalité airways

        Obstacles : hard constraint (1000) → jamais violé
        Airways   : soft constraint        → évité si possible, pas interdit

        FIX : suppression du bloc pts[j] = intermediaires[j-1].copy() qui
        modifiait une copie locale sans effet sur l'optimiseur.
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

        # ── Pénalité airways (soft) ───────────────────────────────────
        # Un waypoint hors airway est pénalisé.
        # Un waypoint dans un airway à contre-sens n'est PAS récompensé.
        penalty_airway = 0.0
        if has_airways:
            for idx_pt, pt in enumerate(pts[1:-1], start=1):
                in_valid_airway = False
                movement = pt - pts[idx_pt - 1]
                norm = np.linalg.norm(movement)
                for airway in airways:
                    if airway.is_point_in_corridor(pt):
                        if norm > 1e-6:
                            dot = np.dot(movement / norm, airway.direction_normalized)
                            if dot >= 0:
                                in_valid_airway = True
                                break
                if not in_valid_airway:
                    penalty_airway += airway_penalty

        # ── Pénalité sens de circulation (barrière quadratique lisse) ─
        # Échantillonnage de chaque segment en 10 points pour détecter
        # les passages à travers un airway même si aucun waypoint n'est dedans.
        penalty_direction = 0.0
        if has_airways:
            for i in range(len(pts) - 1):
                movement = pts[i+1] - pts[i]
                norm = np.linalg.norm(movement)
                if norm < 1e-6:
                    continue
                movement_normalized = movement / norm
                for t in np.linspace(0, 1, 10):
                    sample = pts[i] + t * movement
                    for airway in airways:
                        if airway.is_point_in_corridor(sample):
                            dot = np.dot(movement_normalized, airway.direction_normalized)
                            penalty_direction += max(0.0, -dot) ** 2 * 5000.0

        # ── Coût total ────────────────────────────────────────────────
        cost = (longueur
                + 5.0 * courbure
                + penalty_obs
                + penalty_airway
                + penalty_direction)
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

        if has_airways:
            pts_opt = waypoints_opt[1:-1]
            in_airway = sum(
                1 for pt in pts_opt if is_point_in_any_airway(pt, airways)
            )
            print(f"  Points dans airways      : {in_airway}/{len(pts_opt)} "
                  f"({100*in_airway/max(len(pts_opt),1):.1f}%)")

        print("="*55)

    return waypoints_opt, cost_reduction