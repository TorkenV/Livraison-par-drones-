"""
Lissage de trajectoire par B-Splines cubiques
et calcul du vecteur d'état 10D : [x, y, z, vx, vy, vz, ax, ay, az, m]
Principe :
    - Entrée  : waypoints [x, y, z] issus du RRT
    - Sortie  : trajectoire T(t) de classe C² (position, vitesse, accélération continues)
    - Vecteur d'état : position = T(t), vitesse = T'(t), accélération = T''(t), masse = constante
    
MODIFIÉ pour L-BFGS : Accepte maintenant des temps personnalisés
"""

import numpy as np
from scipy.interpolate import splprep, splev


# ------------------------------------------------------------------ #
# Lissage B-Spline                                                     #
# ------------------------------------------------------------------ #

def smooth_path(waypoints, num_points=200):
    """
    Lisse une liste de waypoints avec une B-Spline cubique (C²)

    Args:
        waypoints  (list[np.array]) : points [x, y, z] issus du RRT
        num_points (int)            : nombre de points sur la trajectoire lissée

    Returns:
        np.array: shape (num_points, 3) — trajectoire lissée [x, y, z]
    """

    waypoints = np.array(waypoints)  # shape (N, 3)

    x = waypoints[:, 0]
    y = waypoints[:, 1]
    z = waypoints[:, 2]

    # Degré de la spline : cubique (k=3) → continuité C²
    k = min(3, len(waypoints) - 1)

    # Calculer la B-Spline (s=0 : passe exactement par les waypoints)
    tck, u = splprep([x, y, z], s=0, k=k)

    # Évaluer la spline sur num_points points uniformes
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new, z_new = splev(u_new, tck)

    return np.column_stack([x_new, y_new, z_new]), tck


# ------------------------------------------------------------------ #
# Calcul du vecteur d'état 10D                                         #
# ------------------------------------------------------------------ #

def compute_state_vector(tck, mass, cruise_speed, num_points=200):
    """
    Calcule le vecteur d'état 10D à chaque instant t :
        position     = T(t)
        vitesse      = T'(t)
        accélération = T''(t)
        masse        = constante

    La paramétrisation temporelle est faite en répartissant
    les points selon la longueur d'arc (vitesse approximativement constante)

    Args:
        tck          : descripteur B-Spline (issu de smooth_path)
        mass         (float) : masse du drone en kg (constante)
        cruise_speed (float) : vitesse de croisière en m/s
        num_points   (int)   : nombre de points dans le vecteur d'état

    Returns:
        states (np.array) : shape (num_points, 10)
                            colonnes : [x, y, z, vx, vy, vz, ax, ay, az, m]
        times  (np.array) : shape (num_points,) — temps en secondes
    """

    # --- 1. Évaluer position, vitesse, accélération sur la spline ---

    u = np.linspace(0, 1, num_points)

    # Position T(t)
    pos = np.array(splev(u, tck, der=0)).T       # shape (N, 3)

    # Dérivée première T'(u) par rapport au paramètre u
    dpos_du = np.array(splev(u, tck, der=1)).T   # shape (N, 3)

    # Dérivée seconde T''(u) par rapport au paramètre u
    d2pos_du2 = np.array(splev(u, tck, der=2)).T # shape (N, 3)

    # --- 2. Paramétrisation temporelle par longueur d'arc ---
    # On veut que ||v(t)|| ≈ cruise_speed
    # ||v(t)|| = ||dpos/dt|| = ||dpos/du|| * |du/dt|
    # donc |du/dt| = cruise_speed / ||dpos/du||

    arc_speeds = np.linalg.norm(dpos_du, axis=1)  # ||dpos/du|| à chaque point
    arc_speeds = np.maximum(arc_speeds, 1e-6)      # éviter division par zéro

    # dt/du = ||dpos/du|| / cruise_speed
    dt_du = arc_speeds / cruise_speed

    # Intégrer pour obtenir le temps à chaque point
    du = 1.0 / (num_points - 1)
    times = np.zeros(num_points)
    for i in range(1, num_points):
        times[i] = times[i-1] + dt_du[i] * du

    # --- 3. Calcul vitesse et accélération dans le repère physique ---
    # v(t) = dpos/du * du/dt = dpos/du * (cruise_speed / ||dpos/du||)

    du_dt = cruise_speed / arc_speeds              # scalaire à chaque point

    velocity     = dpos_du   * du_dt[:, np.newaxis]   # shape (N, 3)
    acceleration = d2pos_du2 * (du_dt[:, np.newaxis])**2  # shape (N, 3)

    # --- 4. Assembler le vecteur d'état 10D ---
    masses = np.full((num_points, 1), mass)        # masse constante

    states = np.hstack([pos, velocity, acceleration, masses])  # shape (N, 10)

    return states, times


def compute_state_vector_with_times(tck, mass, waypoint_times, num_points=200):
    """
    NOUVEAU : Calcule le vecteur d'état 10D avec des temps personnalisés (depuis L-BFGS)
    
    Args:
        tck : descripteur B-Spline
        mass : masse du drone
        waypoint_times : temps d'arrivée à chaque waypoint (depuis L-BFGS)
        num_points : nombre de points
    
    Returns:
        states, times
    """
    
    # Créer une interpolation temporelle sur toute la trajectoire
    T_total = waypoint_times[-1]
    times = np.linspace(0, T_total, num_points)
    
    # Mapper les temps sur le paramètre u de la spline
    # On suppose une correspondance linéaire simple
    u = np.linspace(0, 1, num_points)
    
    # Position
    pos = np.array(splev(u, tck, der=0)).T
    
    # Dérivées par rapport à u
    dpos_du = np.array(splev(u, tck, der=1)).T
    d2pos_du2 = np.array(splev(u, tck, der=2)).T
    
    # Calculer du/dt à partir des temps fournis
    dt = times[1] - times[0]
    du_val = u[1] - u[0]
    du_dt = du_val / dt
    
    # Vitesse et accélération
    velocity = dpos_du * du_dt
    acceleration = d2pos_du2 * (du_dt ** 2)
    
    # Masse constante
    masses = np.full((num_points, 1), mass)
    
    states = np.hstack([pos, velocity, acceleration, masses])
    
    return states, times


# ------------------------------------------------------------------ #
# Fonction principale : waypoints → vecteur d'état complet             #
# ------------------------------------------------------------------ #

def build_trajectory(waypoints, mass, cruise_speed, times=None, num_points=200):
    """
    Pipeline complet : waypoints RRT → trajectoire lissée → vecteur d'état 10D

    Args:
        waypoints    (list[np.array]) : points [x, y, z] issus du RRT
        mass         (float)          : masse du drone (kg)
        cruise_speed (float)          : vitesse de croisière (m/s)
        times        (array)          : NOUVEAU - temps optimaux de L-BFGS (optionnel)
        num_points   (int)            : nombre de points dans la trajectoire

    Returns:
        states       (np.array) : shape (num_points, 10) — vecteur d'état
        times_out    (np.array) : shape (num_points,)    — temps (s)
        smooth       (np.array) : shape (num_points, 3)  — trajectoire lissée
    """

    print(f"  Spline : {len(waypoints)} waypoints → {num_points} points")

    # 1. Lisser les waypoints avec B-Spline cubique
    smooth, tck = smooth_path(waypoints, num_points)

    # 2. Calculer le vecteur d'état 10D
    if times is None:
        # Utiliser vitesse constante (comportement par défaut)
        states, times_out = compute_state_vector(tck, mass, cruise_speed, num_points)
        print(f"  Spline : utilisation vitesse constante ({cruise_speed} m/s)")
    else:
        # Utiliser les temps optimaux fournis par L-BFGS
        states, times_out = compute_state_vector_with_times(
            tck, mass, times, num_points
        )
        print(f"  Spline : utilisation temps optimaux L-BFGS")

    print(f"  Spline : durée totale = {times_out[-1]:.2f} s")
    print(f"  Spline : vecteur d'état shape = {states.shape}")

    return states, times_out, smooth


# ------------------------------------------------------------------ #
# Test du module                                                        #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from load_map import Map
    from rrt import rrt

    print("="*55)
    print("  TEST DU MODULE SPLINES")
    print("="*55)

    for map_id in range(1, 5):
        print(f"\n--- Map {map_id} ---")

        # Charger la map
        m = Map()
        m.load_from_xml(f"../data/maps/map{map_id}.xml")

        # Planifier avec RRT
        path = rrt(
            start         = m.start,
            goal          = m.goal,
            obstacles     = m.obstacles,
            safety_margin = 10.0,
            step_size     = 10.0,
            max_iter      = 5000,
            goal_radius   = 15.0
        )

        if path is None:
            print("  RRT échoué, map ignorée")
            continue

        # Lisser et calculer le vecteur d'état
        states, times, smooth = build_trajectory(
            waypoints    = path,
            mass         = 6.0,
            cruise_speed = 10.0,
            num_points   = 200
        )

        # Afficher un résumé
        print(f"  Trajectoire calculée")
        print(f"     Durée        : {times[-1]:.2f} s")
        print(f"     Vitesse moy  : {np.mean(np.linalg.norm(states[:, 3:6], axis=1)):.2f} m/s")
        print(f"     Accel  moy   : {np.mean(np.linalg.norm(states[:, 6:9], axis=1)):.2f} m/s²")
        print(f"     Masse        : {states[0, 9]:.1f} kg (constante)")
        print(f"     Premier état : {states[0].round(2).tolist()}")
        print(f"     Dernier état : {states[-1].round(2).tolist()}")

    print("\n" + "="*55)
    print("  Test terminé")
    print("="*55)