"""
Visualisation 3D des trajectoires de drone
Auteurs: Chadha Raouafi, Basma Farchi, Ahmed Bousbih
Projet: GPS 4D Collaboratif pour VTOL
INSA Toulouse - 2025-2026

Repère ENU (East-North-Up) :
    X = East, Y = North, Z = Up
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation


# ------------------------------------------------------------------ #
# Dessin des obstacles                                                  #
# ------------------------------------------------------------------ #

def draw_obstacle(ax, obstacle, color='red', alpha=0.3):
    """
    Dessine un obstacle rectangulaire 3D

    Args:
        ax       : axes matplotlib 3D
        obstacle : objet Obstacle (depuis load_map)
        color    : couleur de l'obstacle
        alpha    : transparence
    """
    v = obstacle.get_vertices()

    faces = [
        [v[0], v[1], v[2], v[3]],  # bas
        [v[4], v[5], v[6], v[7]],  # haut
        [v[0], v[1], v[5], v[4]],  # avant
        [v[2], v[3], v[7], v[6]],  # arrière
        [v[1], v[2], v[6], v[5]],  # droite
        [v[0], v[3], v[7], v[4]],  # gauche
    ]

    cube = Poly3DCollection(faces, facecolors=color,
                            edgecolors='darkred', linewidths=0.5, alpha=alpha)
    ax.add_collection3d(cube)

    # Nom de l'obstacle au-dessus
    ax.text(obstacle.center[0], obstacle.center[1],
            obstacle.max_bound[2] + 5,
            obstacle.name, fontsize=8, ha='center', color='darkred')


# ------------------------------------------------------------------ #
# Dessin des routes aériennes                                          #
# ------------------------------------------------------------------ #

def draw_airway(ax, airway, color='violet', alpha=0.25):
    """
    Dessine une route aérienne cylindrique en 3D avec une flèche
    indiquant le sens de circulation (waypoint_start -> waypoint_end).
    """
    # 1. Vecteurs perpendiculaires à l'axe
    d = airway.direction_normalized
    if abs(d[2]) < 0.9:
        perp1 = np.cross(d, [0, 0, 1])
    else:
        perp1 = np.cross(d, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(d, perp1)

    radius = airway.width / 2
    theta = np.linspace(0, 2 * np.pi, 30)

    # 2. Disques le long de l'axe
    for t in np.linspace(0, 1, 6):
        center = airway.waypoint_start + t * airway.direction
        cx = center[0] + radius * (np.cos(theta) * perp1[0] + np.sin(theta) * perp2[0])
        cy = center[1] + radius * (np.cos(theta) * perp1[1] + np.sin(theta) * perp2[1])
        cz = center[2] + radius * (np.cos(theta) * perp1[2] + np.sin(theta) * perp2[2])
        verts = [list(zip(cx, cy, cz))]
        poly = Poly3DCollection(verts, facecolors=color,
                                edgecolors=color, linewidths=0.3, alpha=alpha)
        ax.add_collection3d(poly)

    # 3. Fleche de sens au centre de la route
    mid = (airway.waypoint_start + airway.waypoint_end) / 2
    arrow_len = airway.length * 0.35
    arrow_vec = d * arrow_len
    ax.quiver(
        mid[0] - arrow_vec[0] / 2,
        mid[1] - arrow_vec[1] / 2,
        mid[2] - arrow_vec[2] / 2,
        arrow_vec[0], arrow_vec[1], arrow_vec[2],
        color='darkviolet', linewidth=2.5,
        arrow_length_ratio=0.3, zorder=10
    )

    # 4. Nom de la route a l'entree
    ax.text(
        airway.waypoint_start[0], airway.waypoint_start[1],
        airway.waypoint_start[2] + radius + 3,
        airway.name, fontsize=7, ha='center', color='darkviolet'
    )



# ------------------------------------------------------------------ #
# Visualisation statique                                               #
# ------------------------------------------------------------------ #

def plot_trajectory(states, times, map_obj, waypoints=None, smooth=None, title=None):
    """
    Affiche la trajectoire complète en 3D (vue statique)

    Args:
        states    (np.array) : vecteur d'état (N, 10) — [x,y,z,vx,vy,vz,ax,ay,az,m]
        times     (np.array) : temps (N,)
        map_obj              : objet Map (depuis load_map)
        waypoints (list)     : waypoints bruts du RRT (optionnel)
        smooth    (np.array) : trajectoire lissée (optionnel)
        title     (str)      : titre du graphique
    """

    fig = plt.figure(figsize=(14, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # --- Trajectoire lissée ---
    ax.plot(states[:, 0], states[:, 1], states[:, 2],
            'b-', linewidth=2, label='Trajectoire (B-Spline)', zorder=5)

    # --- Waypoints RRT bruts (optionnel) ---
    if waypoints is not None:
        wp = np.array(waypoints)
        ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2],
                   c='orange', s=20, label='Waypoints RRT', zorder=4, alpha=0.6)

    # --- Points départ et arrivée ---
    ax.scatter(*map_obj.start, c='green', s=150, marker='o',
               label='Départ', zorder=10, edgecolors='black')
    ax.scatter(*map_obj.goal,  c='red',   s=150, marker='*',
               label='Arrivée', zorder=10, edgecolors='black')

    # --- Vecteurs vitesse (sous-échantillonnés) ---
    step = max(1, len(states) // 15)
    for i in range(0, len(states), step):
        pos = states[i, 0:3]
        vel = states[i, 3:6]
        v_norm = np.linalg.norm(vel)
        if v_norm > 0.1:
            vel_scaled = vel / v_norm * 12   # longueur fixe pour lisibilité
            ax.quiver(pos[0], pos[1], pos[2],
                      vel_scaled[0], vel_scaled[1], vel_scaled[2],
                      color='lime', linewidth=1.5, arrow_length_ratio=0.3)

    # --- Obstacles ---
    for obs in map_obj.obstacles:
        draw_obstacle(ax, obs)

    # --- Routes aeriennes avec fleche de sens ---
    for airway in map_obj.airways:
        draw_airway(ax, airway)

    # --- Mise en forme ---
    ax.set_xlabel('X - East (m)',  fontsize=11)
    ax.set_ylabel('Y - North (m)', fontsize=11)
    ax.set_zlabel('Z - Up (m)',    fontsize=11)

    if title is None:
        title = f'{map_obj.name}  |  durée: {times[-1]:.1f}s  |  masse: {states[0,9]:.1f}kg'
    ax.set_title(title, fontsize=12, fontweight='bold')

    ax.legend(loc='upper left', fontsize=9)
    ax.view_init(elev=25, azim=45)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
# Visualisation des profils vitesse et accélération                    #
# ------------------------------------------------------------------ #

def plot_profiles(states, times, map_obj):
    """
    Affiche les profils de vitesse et d'accélération en fonction du temps

    Args:
        states  (np.array) : vecteur d'état (N, 10)
        times   (np.array) : temps (N,)
        map_obj            : objet Map
    """

    # Normes vitesse et accélération
    v_norm = np.linalg.norm(states[:, 3:6], axis=1)
    a_norm = np.linalg.norm(states[:, 6:9], axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f'Profils dynamiques — {map_obj.name}', fontsize=13, fontweight='bold')

    # --- Position XYZ ---
    axes[0].plot(times, states[:, 0], label='X (East)',  color='blue')
    axes[0].plot(times, states[:, 1], label='Y (North)', color='green')
    axes[0].plot(times, states[:, 2], label='Z (Up)',    color='red')
    axes[0].set_ylabel('Position (m)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Vitesse ---
    axes[1].plot(times, v_norm, color='lime', linewidth=2, label='||v|| (m/s)')
    axes[1].axhline(y=states[0, 3:6].mean(), color='gray',
                    linestyle='--', alpha=0.5, label='ref')
    axes[1].set_ylabel('Vitesse (m/s)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # --- Accélération ---
    axes[2].plot(times, a_norm, color='orange', linewidth=2, label='||a|| (m/s²)')
    axes[2].set_ylabel('Accélération (m/s²)')
    axes[2].set_xlabel('Temps (s)')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
# Animation 3D                                                         #
# ------------------------------------------------------------------ #

def animate_trajectory(states, times, map_obj, speed_factor=1.0):
    """
    Animation 3D du drone le long de la trajectoire

    Args:
        states       (np.array) : vecteur d'état (N, 10)
        times        (np.array) : temps (N,)
        map_obj                 : objet Map
        speed_factor (float)    : 1.0 = temps réel, 2.0 = x2 plus rapide
    """

    fig = plt.figure(figsize=(14, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # Limites des axes
    margin = 50
    ax.set_xlim(states[:, 0].min() - margin, states[:, 0].max() + margin)
    ax.set_ylim(states[:, 1].min() - margin, states[:, 1].max() + margin)
    ax.set_zlim(0, max(states[:, 2].max() + margin, 100))

    ax.set_xlabel('X - East (m)',  fontsize=11)
    ax.set_ylabel('Y - North (m)', fontsize=11)
    ax.set_zlabel('Z - Up (m)',    fontsize=11)
    ax.set_title(f'Animation — {map_obj.name}', fontsize=12, fontweight='bold')
    ax.view_init(elev=25, azim=45)
    ax.grid(True, alpha=0.3)

    # Obstacles (fixes)
    for obs in map_obj.obstacles:
        draw_obstacle(ax, obs)

    # Routes aeriennes avec fleche de sens (fixes)
    for airway in map_obj.airways:
        draw_airway(ax, airway)

    # Trajectoire complète en pointillés (fixe)
    ax.plot(states[:, 0], states[:, 1], states[:, 2],
            'b--', linewidth=1, alpha=0.3, label='Trajectoire planifiée')

    # Départ et arrivée (fixes)
    ax.scatter(*map_obj.start, c='green', s=150, marker='o',
               label='Départ', zorder=10, edgecolors='black')
    ax.scatter(*map_obj.goal,  c='red',   s=150, marker='*',
               label='Arrivée', zorder=10, edgecolors='black')

    # Éléments animés
    path_line,   = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.8)
    drone_point  = ax.scatter([], [], [], c='blue', s=300,
                              edgecolors='black', zorder=10)

    # Texte info
    info = ax.text2D(0.02, 0.97, '', transform=ax.transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85))

    vel_quiver  = [None]

    def update(frame):
        # Trajectoire parcourue
        path_line.set_data(states[:frame+1, 0], states[:frame+1, 1])
        path_line.set_3d_properties(states[:frame+1, 2])

        # Position drone
        drone_point._offsets3d = (
            [states[frame, 0]],
            [states[frame, 1]],
            [states[frame, 2]]
        )

        # Vecteur vitesse
        if vel_quiver[0] is not None:
            vel_quiver[0].remove()

        vel = states[frame, 3:6]
        v_norm = np.linalg.norm(vel)
        if v_norm > 0.1:
            vel_scaled = vel / v_norm * 15
            vel_quiver[0] = ax.quiver(
                states[frame, 0], states[frame, 1], states[frame, 2],
                vel_scaled[0], vel_scaled[1], vel_scaled[2],
                color='lime', linewidth=2, arrow_length_ratio=0.3
            )

        # Texte info
        info.set_text(
            f"t = {times[frame]:.1f} s\n"
            f"Position : ({states[frame,0]:.1f}, {states[frame,1]:.1f}, {states[frame,2]:.1f}) m\n"
            f"||v|| = {v_norm:.2f} m/s\n"
            f"masse = {states[frame, 9]:.1f} kg"
        )

        return path_line, drone_point, info

    # Calcul intervalle
    dt_avg   = (times[-1] - times[0]) / len(times)
    interval = max(20, int(dt_avg * 1000 / speed_factor))

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval, blit=False, repeat=True)

    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()

    return anim



if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from load_map import Map
    from rrt import rrt
    from splines import build_trajectory

    print("="*55)
    print("  TEST DU MODULE VISUALIZATION")
    print("="*55)

    # Map a visualiser (Exemple : 2,3,4)
    MAP_ID = int(sys.argv[1])

    print(f"\nChargement de la map {MAP_ID}...")
    m = Map()
    m.load_from_xml(f"..\\data\\map{MAP_ID}.xml")

    print("\nPlanification RRT")
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
        print("RRT échoué")
        exit()

    print("\nCalcul de la trajectoire")
    states, times, smooth = build_trajectory(
        waypoints    = path,
        mass         = 6.0,
        cruise_speed = 10.0,
        num_points   = 200
    )

    # 1. Vue statique avec vecteurs vitesse
    print("\n1. Affichage statique...")
    plot_trajectory(states, times, m, waypoints=path)

    # 2. Profils dynamiques
    print("2. Profils vitesse/accélération...")
    plot_profiles(states, times, m)

    # 3. Animation
    print("3. Animation 3D...")
    animate_trajectory(states, times, m, speed_factor=2.0)