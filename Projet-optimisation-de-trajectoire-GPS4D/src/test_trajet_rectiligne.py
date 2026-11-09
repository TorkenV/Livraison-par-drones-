import numpy as np
from load_drone import Drone
from load_map   import Map
from splines    import build_trajectory
from visualization import plot_trajectory, plot_profiles, animate_trajectory


# ------------------------------------------------------------------ #
# Génération de trajectoire rectiligne directe                         #
# ------------------------------------------------------------------ #

def generate_straight_path(start, goal, num_waypoints=10):
    """
    Génère une trajectoire rectiligne parfaite entre start et goal

    Args:
        start         (array) : point de départ [x, y, z]
        goal          (array) : point d'arrivée [x, y, z]
        num_waypoints (int)   : nombre de waypoints intermédiaires

    Returns:
        list[np.array] : waypoints uniformément espacés
    """
    start = np.array(start)
    goal  = np.array(goal)

    waypoints = []
    for i in range(num_waypoints):
        t = i / (num_waypoints - 1)  # de 0 à 1
        point = start + t * (goal - start)
        waypoints.append(point)

    return waypoints


# ------------------------------------------------------------------ #
# Programme principal                                                   #
# ------------------------------------------------------------------ #

def test_trajectoire_rectiligne():
    """
    Test de trajectoire rectiligne sur map1
    """

    print("\n" + "="*70)
    print("  TEST TRAJECTOIRE RECTILIGNE (sans RRT)")
    print("  Map 1 uniquement - ligne droite A → B")
    print("="*70)

    # ── 1. Charger le drone ──────────────────────────────────────────
    print("\n[1/4] Chargement du drone...")
    drone = Drone()
    drone.load_from_xml("C:\\Users\\Dr console\\Desktop\\INSA\\4A\\PIR\\Projet-optimisation-de-trajectoire-GPS4D\\data\\drone1.xml")

    # ── 2. Charger la map1 ───────────────────────────────────────────
    print("\n[2/4] Chargement de la map1...")
    m = Map()
    m.load_from_xml("C:\\Users\\Dr console\\Desktop\\INSA\\4A\\PIR\\Projet-optimisation-de-trajectoire-GPS4D\\data\\map1.xml")

    if len(m.obstacles) > 0:
        print("⚠️  Attention : map1 contient des obstacles, ce test suppose une ligne droite")

    # ── 3. Générer trajectoire rectiligne (SANS RRT) ─────────────────
    print("\n[3/4] Génération de la trajectoire rectiligne...")
    print(f"   Départ  : {m.start}")
    print(f"   Arrivée : {m.goal}")

    # Créer une ligne droite avec seulement quelques waypoints
    path = generate_straight_path(m.start, m.goal, num_waypoints=10)

    print(f"   ✅ Trajectoire rectiligne : {len(path)} waypoints")

    # ── 4. B-Splines + vecteur d'état 10D ───────────────────────────
    print("\n[4/4] Calcul du vecteur d'état 10D (B-Splines)...")
    states, times, smooth = build_trajectory(
        waypoints    = path,
        mass         = drone.mass_loaded,
        cruise_speed = drone.cruise_speed,
        num_points   = 200
    )

    # ── Résumé ───────────────────────────────────────────────────────
    distance_directe = np.linalg.norm(m.goal - m.start)

    print("\n" + "="*70)
    print("  RÉSUMÉ - TRAJECTOIRE RECTILIGNE")
    print("="*70)
    print(f"  Type           : Ligne droite (sans RRT)")
    print(f"  Distance       : {distance_directe:.2f} m")
    print(f"  Waypoints      : {len(path)}")
    print(f"  Points spline  : {len(states)}")
    print(f"  Durée de vol   : {times[-1]:.2f} s")
    print(f"  Vitesse moy    : {np.mean(np.linalg.norm(states[:,3:6], axis=1)):.2f} m/s")
    print(f"  Vitesse théo   : {distance_directe / times[-1]:.2f} m/s")
    print(f"  Masse          : {states[0,9]:.1f} kg (constante)")

    # Analyse du vecteur d'état
    print(f"\n  Vecteur d'état 10D:")
    print(f"    Position     : x ∈ [{states[:,0].min():.1f}, {states[:,0].max():.1f}]")
    print(f"                   y ∈ [{states[:,1].min():.1f}, {states[:,1].max():.1f}]")
    print(f"                   z ∈ [{states[:,2].min():.1f}, {states[:,2].max():.1f}]")

    v_norms = np.linalg.norm(states[:,3:6], axis=1)
    print(f"    Vitesse      : ||v|| ∈ [{v_norms.min():.2f}, {v_norms.max():.2f}] m/s")

    a_norms = np.linalg.norm(states[:,6:9], axis=1)
    print(f"    Accélération : ||a|| ∈ [{a_norms.min():.2f}, {a_norms.max():.2f}] m/s²")
    print(f"    Accél. moy   : {a_norms.mean():.2f} m/s²")

    print("="*70)

    # ── Visualisation ────────────────────────────────────────────────
    print("\n📊 Affichage des graphiques...")

    # Vue statique 3D
    plot_trajectory(
        states    = states,
        times     = times,
        map_obj   = m,
        waypoints = path,
        title     = f"Trajectoire rectiligne — {m.name}"
    )

    # Profils vitesse / accélération
    plot_profiles(states, times, m)

    # Animation 3D
    print("\n🎬 Lancement de l'animation...")
    animate_trajectory(states, times, m, speed_factor=2.0)

    print("\n✅ Test terminé")


# ------------------------------------------------------------------ #
# Point d'entrée                                                        #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    test_trajectoire_rectiligne()