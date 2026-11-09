import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from load_drone import Drone
from load_map   import Map
from rrt import rrt
from splines    import smooth_path, build_trajectory
from lbfgs_optimizer2 import optimize_trajectory_lbfgs2 as optimize_trajectory_lbfgs
from visualization import plot_profiles, animate_trajectory

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)  
# ------------------------------------------------------------------ #
# Visualisation des routes aériennes                                  #
# ------------------------------------------------------------------ #

def draw_airway_cylinder(ax, airway, color='lightgreen', alpha=0.15):
    start = airway.waypoint_start
    end = airway.waypoint_end
    radius = airway.width / 2

    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return

    direction_normalized = direction / length

    if abs(direction_normalized[2]) < 0.9:
        perp1 = np.cross(direction_normalized, [0, 0, 1])
    else:
        perp1 = np.cross(direction_normalized, [1, 0, 0])

    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction_normalized, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    n_segments = 20
    theta = np.linspace(0, 2*np.pi, n_segments)

    circle_points = []
    for t in theta:
        offset = radius * (np.cos(t) * perp1 + np.sin(t) * perp2)
        circle_points.append(offset)
    circle_points = np.array(circle_points)

    for i in range(n_segments - 1):
        p1_start = start + circle_points[i]
        p2_start = start + circle_points[i+1]
        p1_end   = end   + circle_points[i]
        p2_end   = end   + circle_points[i+1]

        face = [p1_start, p2_start, p2_end, p1_end]
        poly = Poly3DCollection([face], facecolors=color,
                                edgecolors='none', alpha=alpha)
        ax.add_collection3d(poly)

    start_circle = [start + cp for cp in circle_points]
    end_circle   = [end   + cp for cp in circle_points]

    poly_start = Poly3DCollection([start_circle], facecolors=color,
                                  edgecolors='darkgreen', alpha=alpha*2, linewidths=1)
    poly_end   = Poly3DCollection([end_circle],   facecolors=color,
                                  edgecolors='darkgreen', alpha=alpha*2, linewidths=1)
    ax.add_collection3d(poly_start)
    ax.add_collection3d(poly_end)

    ax.plot([start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color='darkblue', linewidth=2, linestyle='--', alpha=0.6)

    # Fleche indiquant le sens de circulation (start -> end)
    mid_point = (start + end) / 2
    arrow_len = length * 0.15
    arrow_vec = direction_normalized * arrow_len
    ax.quiver(
        mid_point[0] - arrow_vec[0] / 2,
        mid_point[1] - arrow_vec[1] / 2,
        mid_point[2] - arrow_vec[2] / 2,
        arrow_vec[0], arrow_vec[1], arrow_vec[2],
        color='darkgreen', linewidth=3.0,
        arrow_length_ratio=0.4, zorder=10
    )

    # Nom de la route a l'entree du couloir
    ax.text(start[0], start[1], start[2] + radius + 3,
            airway.name, fontsize=7, ha='center', color='darkblue')

def draw_obstacle(ax, obstacle, color='blue', alpha=0.3):
    """Dessine un obstacle rectangulaire 3D"""
    v = obstacle.get_vertices()

    faces = [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[0], v[3], v[7], v[4]],
    ]

    cube = Poly3DCollection(faces, facecolors=color,
                            edgecolors='darkblue', linewidths=0.5, alpha=alpha)
    ax.add_collection3d(cube)

    ax.text(obstacle.center[0], obstacle.center[1],
            obstacle.max_bound[2] + 5,
            obstacle.name, fontsize=8, ha='center', color='darkred')


def plot_trajectory_3d(states, times, map_obj, waypoints_rrt=None,
                       waypoints_opt=None, title=None):
    """
    Affiche la trajectoire complète en 3D avec les routes aériennes

    Args:
        states         (np.array) : vecteur d'état (N, 10)
        times          (np.array) : temps (N,)
        map_obj                   : objet Map
        waypoints_rrt  (list)     : waypoints bruts du RRT (optionnel)
        waypoints_opt  (np.array) : waypoints après L-BFGS (optionnel)
        title          (str)      : titre du graphique
    """

    fig = plt.figure(figsize=(16, 10))
    ax  = fig.add_subplot(111, projection='3d')

    # Waypoints RRT bruts
    if waypoints_rrt is not None:
        wp = np.array(waypoints_rrt)
        ax.plot(wp[:, 0], wp[:, 1], wp[:, 2],
                'o--', color='gray', linewidth=1, markersize=3,
                label='Waypoints RRT', zorder=3, alpha=0.5)

    # Waypoints après L-BFGS
    if waypoints_opt is not None:
        wo = np.array(waypoints_opt)
        ax.plot(wo[:, 0], wo[:, 1], wo[:, 2],
                's--', color='orange', linewidth=1.5, markersize=4,
                label='Waypoints L-BFGS', zorder=4, alpha=0.7)

    # Trajectoire finale (B-Spline sur waypoints optimisés)
    ax.plot(states[:, 0], states[:, 1], states[:, 2],
            color='violet', linewidth=2.5,
            label='Trajectoire finale (B-Spline)', zorder=5)

    # Départ et arrivée
    ax.scatter(*map_obj.start, c='green', s=150, marker='o',
               label='Départ', zorder=10, edgecolors='black')
    ax.scatter(*map_obj.goal,  c='green',   s=150, marker='*',
               label='Arrivée', zorder=10, edgecolors='black')

    # Vecteurs vitesse (sous-échantillonnés)
    step = max(1, len(states) // 15)
    for i in range(0, len(states), step):
        pos = states[i, 0:3]
        vel = states[i, 3:6]
        v_norm = np.linalg.norm(vel)
        if v_norm > 0.1:
            vel_scaled = vel / v_norm * 12
            ax.quiver(pos[0], pos[1], pos[2],
                      vel_scaled[0], vel_scaled[1], vel_scaled[2],
                      color='lime', linewidth=1.5, arrow_length_ratio=0.3)

    # Vecteurs accélération (sous-échantillonnés)
    for i in range(0, len(states), step):
        pos = states[i, 0:3]
        acc = states[i, 6:9]
        a_norm = np.linalg.norm(acc)
        if a_norm > 0.5:
            acc_scaled = acc / a_norm * 8
            ax.quiver(pos[0], pos[1], pos[2],
                      acc_scaled[0], acc_scaled[1], acc_scaled[2],
                      color='orange', linewidth=1.2, arrow_length_ratio=0.3)

    # Obstacles
    for obs in map_obj.obstacles:
        draw_obstacle(ax, obs)

    # Mise en forme
    ax.set_xlabel('X - East (m)',  fontsize=11)
    ax.set_ylabel('Y - North (m)', fontsize=11)
    ax.set_zlabel('Z - Up (m)',    fontsize=11)

    if title is None:
        title = f'{map_obj.name}  |  durée: {times[-1]:.1f}s'
    ax.set_title(title, fontsize=13, fontweight='bold')

    ax.legend(loc='upper left', fontsize=9)
    ax.view_init(elev=25, azim=45)

    # Forcer le meme rapport d'echelle sur les 3 axes
    all_x = [map_obj.start[0], map_obj.goal[0]] + [s[0] for s in states]
    all_y = [map_obj.start[1], map_obj.goal[1]] + [s[1] for s in states]
    all_z = [map_obj.start[2], map_obj.goal[2]] + [s[2] for s in states]
    max_range = max(
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z)
    ) / 2.0
    mid_x = (max(all_x) + min(all_x)) / 2
    mid_y = (max(all_y) + min(all_y)) / 2
    mid_z = (max(all_z) + min(all_z)) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

DRONE_FILE = "..\\data\\drone1.xml"

MAP_FILES = {
    1: "..\\data\\map1.xml",
    2: "..\\data\\map2.xml",
    3: "..\\data\\map3.xml",
    4: "..\\data\\map4.xml",
    5: "..\\data\\map5.xml",
    6: "..\\data\\map6.xml",
    7 : "..\\data\\map7.xml",
    8 : "..\\data\\map8.xml",
    9 : "..\\data\\map9.xml"
}


# ------------------------------------------------------------------ #
# Programme principal                                                  #
# ------------------------------------------------------------------ #

def run(map_id=1, show_waypoints=True, animate=False):
    """
    Pipeline complet : chargement → RRT → B-Splines → L-BFGS → B-Splines → visualisation

    Args:
        map_id         (int)  : scénario à exécuter (1 à 6)
        show_waypoints (bool) : afficher les waypoints RRT bruts et L-BFGS
        animate        (bool) : lancer l'animation 3D
    """

    print("\n" + "="*55)
    print(f"  SIMULATION — MAP {map_id} — METHOD 2 : RRT + L-BFGS + B-Splines")
    print("="*55)

    # ── 1. Charger le drone ──────────────────────────────────────────
    print("\n[1/6] Chargement du drone...")
    drone = Drone()
    drone.load_from_xml(DRONE_FILE)

    # ── 2. Charger la map ────────────────────────────────────────────
    print("\n[2/6] Chargement de la map...")
    if map_id not in MAP_FILES:
        print(f"❌ Map {map_id} invalide. Choisir entre 1 et 9.")
        return

    m = Map()
    m.load_from_xml(MAP_FILES[map_id])

    # ── 3. Planification RRT ─────────────────────────────────────────
    print("\n[3/6] Planification RRT...")
    path_rrt = rrt(
        start         = m.start,
        goal          = m.goal,
        obstacles     = m.obstacles,
        safety_margin = drone.max_deceleration,
        step_size     = 10.0,
        max_iter      = 1000,
        goal_radius   = 15.0
    )

    if path_rrt is None:
        print("❌ RRT : aucun chemin trouvé. Augmentez max_iter ou ajustez les bornes.")
        return

    # ── 4. B-Splines (premier lissage) ──────────────────────────────
    # On génère 50 points lissés qui serviront d'entrée à L-BFGS
    # (moins de variables = optimisation plus rapide et stable)
    print("\n[4/6] Premier lissage B-Splines...")
    smooth_pts, _ = smooth_path(path_rrt, num_points=50)
    print(f"  B-Splines : {len(path_rrt)} waypoints RRT → 50 points lissés")

    # ── 5. Optimisation L-BFGS (positions) ──────────────────────────
    print("\n[5/6] Optimisation L-BFGS (positions)...")
    path_optimise, cost_reduction = optimize_trajectory_lbfgs(
        waypoints_rrt = smooth_pts.tolist(),
        drone         = drone,
        obstacles     = m.obstacles,
        verbose       = True
    )

    # ── 6. B-Splines final + vecteur d'état 10D ─────────────────────
    print("\n[6/6] Trajectoire finale (B-Splines sur waypoints optimisés)...")
    states, times, smooth_final = build_trajectory(
        waypoints    = path_optimise,
        mass         = drone.mass_loaded,
        cruise_speed = drone.cruise_speed,
        num_points   = 200
    )

    # ── Résumé ───────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  RÉSUMÉ")
    print("="*55)
    print(f"  Map              : {m.name}")
    print(f"  Drone            : {drone.name}")
    print(f"  Obstacles        : {len(m.obstacles)}")
    print(f"  Routes aériennes : {len(m.airways)}")
    print(f"  Waypoints RRT    : {len(path_rrt)}")
    print(f"  Après L-BFGS     : {len(path_optimise)} points")
    print(f"  Points finaux    : {len(states)}")
    print(f"  Optimisation     : L-BFGS (-{cost_reduction:.1f}% coût)")
    print(f"  Durée de vol     : {times[-1]:.2f} s")
    print(f"  Vitesse moy      : {np.mean(np.linalg.norm(states[:,3:6], axis=1)):.2f} m/s")
    print(f"  Masse            : {states[0,9]:.1f} kg (constante)")
    print("="*55)

    # ── Visualisation ────────────────────────────────────────────────
    plot_trajectory_3d(
        states        = states,
        times         = times,
        map_obj       = m,
        waypoints_rrt = path_rrt        if show_waypoints else None,
        waypoints_opt = path_optimise   if show_waypoints else None
    )

    # Profils vitesse / accélération
    plot_profiles(states, times, m)

    # Animation (optionnelle)
    if animate:
        animate_trajectory(states, times, m, speed_factor=2.0)


if __name__ == "__main__":

    MAP_ID         = int(sys.argv[1])   # 1, 2, 3, 4, 5, 6, 7,8,9
    SHOW_WAYPOINTS = True
    ANIMATE        = False

    run(map_id=MAP_ID, show_waypoints=SHOW_WAYPOINTS, animate=ANIMATE)