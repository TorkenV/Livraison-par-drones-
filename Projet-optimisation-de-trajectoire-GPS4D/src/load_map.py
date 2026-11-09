import numpy as np
import xml.etree.ElementTree as ET

# On définit la classe des routes aériennes 
class Airway:

    def __init__(self, name, waypoint_start, waypoint_end, width, 
                 altitude_min, altitude_max, airway_type='mandatory'):

        self.name = name
        self.waypoint_start = np.array(waypoint_start, dtype=float)
        self.waypoint_end = np.array(waypoint_end, dtype=float)
        self.width = width
        self.altitude_min = altitude_min
        self.altitude_max = altitude_max
        self.airway_type = airway_type
        
        # Calculer vecteur directeur et longueur
        self.direction = self.waypoint_end - self.waypoint_start
        self.length = np.linalg.norm(self.direction)
        if self.length > 0:
            self.direction_normalized = self.direction / self.length
        else:
            self.direction_normalized = np.array([0, 0, 0])

    def is_point_in_corridor(self, point, tolerance=0.0):

        point = np.array(point, dtype=float)

        # 1. Projection du point sur l'axe de la route
        vector_to_point = point - self.waypoint_start
        projection_length = np.dot(vector_to_point, self.direction_normalized)

        # Le point doit se projeter dans les limites du segment
        if projection_length < 0 or projection_length > self.length:
            return False

        # 2. Distance perpendiculaire à l'axe (cylindre pur, pas de contrainte altitude)
        projection_point = self.waypoint_start + projection_length * self.direction_normalized
        perpendicular_distance = np.linalg.norm(point - projection_point)

        return perpendicular_distance <= (self.width / 2 + tolerance)

    def get_closest_point_on_axis(self, point):
        point = np.array(point, dtype=float)
        vector_to_point = point - self.waypoint_start
        projection_length = np.dot(vector_to_point, self.direction_normalized)
        projection_length = np.clip(projection_length, 0, self.length)
        return self.waypoint_start + projection_length * self.direction_normalized

    def __repr__(self):
        return (f"Airway('{self.name}', type='{self.airway_type}', "
                f"start={self.waypoint_start.tolist()}, "
                f"end={self.waypoint_end.tolist()}, "
                f"width={self.width}m)")


# On définit la classe des obstacles                                                     
class Obstacle:

    def __init__(self, center, dimensions, name="Obstacle"):
        self.name = name
        self.center = np.array(center, dtype=float)
        self.dimensions = np.array(dimensions, dtype=float)
        
        half_dims = self.dimensions / 2
        self.min_bound = self.center - half_dims
        self.max_bound = self.center + half_dims

    def is_collision(self, point, safety_margin=0.0):
        point = np.array(point)
        return np.all(point >= self.min_bound - safety_margin) and \
               np.all(point <= self.max_bound + safety_margin)

    def get_vertices(self):
        x_min, y_min, z_min = self.min_bound
        x_max, y_max, z_max = self.max_bound
        
        return np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ])

    def __repr__(self):
        return (f"Obstacle('{self.name}', "
                f"center={self.center.tolist()}, "
                f"dim={self.dimensions.tolist()})")


# Classe principale de la map 
class Map:

    def __init__(self):
        self.name = "Map"
        self.description = ""
        self.scenario_id = 1
        
        self.start = np.array([0.0, 0.0, 0.0])
        self.goal = np.array([500.0, 0.0, 0.0])
        
        self.obstacles = []
        self.airways = []

    def load_from_xml(self, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()

        self.name = self._get_text(root, 'n', 'Map')
        self.description = self._get_text(root, 'description', '')
        self.scenario_id = self._get_int(root, 'scenario_id', 1)

        mission = root.find('mission')
        if mission is not None:
            start_elem = mission.find('start')
            goal_elem = mission.find('goal')
            
            if start_elem is not None:
                self.start = np.array([
                    self._get_float(start_elem, 'x', 0.0),
                    self._get_float(start_elem, 'y', 0.0),
                    self._get_float(start_elem, 'z', 0.0)
                ])
            
            if goal_elem is not None:
                self.goal = np.array([
                    self._get_float(goal_elem, 'x', 500.0),
                    self._get_float(goal_elem, 'y', 0.0),
                    self._get_float(goal_elem, 'z', 0.0)
                ])

        # Routes aériennes
        airways_elem = root.find('airways')
        if airways_elem is not None:
            for airway_elem in airways_elem.findall('airway'):
                name = self._get_text(airway_elem, 'n', 'Route')
                airway_type = self._get_text(airway_elem, 'type', 'mandatory')
                
                start_wp = airway_elem.find('waypoint_start')
                end_wp = airway_elem.find('waypoint_end')
                
                waypoint_start = np.array([
                    self._get_float(start_wp, 'x', 0.0),
                    self._get_float(start_wp, 'y', 0.0),
                    self._get_float(start_wp, 'z', 0.0)
                ])
                
                waypoint_end = np.array([
                    self._get_float(end_wp, 'x', 0.0),
                    self._get_float(end_wp, 'y', 0.0),
                    self._get_float(end_wp, 'z', 0.0)
                ])
                
                width = self._get_float(airway_elem, 'width', 30.0)
                alt_min = self._get_float(airway_elem, 'altitude_min', 0.0)
                alt_max = self._get_float(airway_elem, 'altitude_max', 120.0)
                
                airway = Airway(name, waypoint_start, waypoint_end, 
                               width, alt_min, alt_max, airway_type)
                self.airways.append(airway)

        # Obstacles
        obstacles_elem = root.find('obstacles')
        if obstacles_elem is not None:
            for obs_elem in obstacles_elem.findall('obstacle'):
                name = self._get_text(obs_elem, 'n', 'Obstacle')
                
                center_elem = obs_elem.find('center')
                dims_elem = obs_elem.find('dimensions')
                min_bound_elem = obs_elem.find('min_bound')
                max_bound_elem = obs_elem.find('max_bound')
                
                if center_elem is not None and dims_elem is not None:
                    center = np.array([
                        self._get_float(center_elem, 'x', 0.0),
                        self._get_float(center_elem, 'y', 0.0),
                        self._get_float(center_elem, 'z', 0.0)
                    ])
                    dimensions = np.array([
                        self._get_float(dims_elem, 'width', 10.0),
                        self._get_float(dims_elem, 'depth', 10.0),
                        self._get_float(dims_elem, 'height', 10.0)
                    ])
                    obstacle = Obstacle(center, dimensions, name)
                    
                elif min_bound_elem is not None and max_bound_elem is not None:
                    min_bound = np.array([
                        self._get_float(min_bound_elem, 'x', 0.0),
                        self._get_float(min_bound_elem, 'y', 0.0),
                        self._get_float(min_bound_elem, 'z', 0.0)
                    ])
                    max_bound = np.array([
                        self._get_float(max_bound_elem, 'x', 10.0),
                        self._get_float(max_bound_elem, 'y', 10.0),
                        self._get_float(max_bound_elem, 'z', 10.0)
                    ])
                    center = (min_bound + max_bound) / 2
                    dimensions = max_bound - min_bound
                    obstacle = Obstacle(center, dimensions, name)
                else:
                    print(f"  ⚠️  Obstacle '{name}' : format invalide (ignoré)")
                    continue
                
                self.obstacles.append(obstacle)

        # Affichage résumé
        dist = np.linalg.norm(self.goal - self.start)
        
        print("="*55)
        print(f"  Map chargée  : {self.name}")
        print(f"  Fichier      : {filepath}")
        print(f"  Scénario     : {self.scenario_id}")
        print("="*55)
        print(f"  Départ  (ENU) : {self.start.tolist()}")
        print(f"  Arrivée (ENU) : {self.goal.tolist()}")
        print(f"  Distance directe : {dist:.2f} m")
        
        print(f"  Obstacles : {len(self.obstacles)}")
        for obs in self.obstacles:
            print(f"    - {obs}")
        
        print(f"  Routes aériennes : {len(self.airways)}")
        for airway in self.airways:
            print(f"    - {airway.name} : >>------ {airway.name} ------>>")
        
        print("="*55)

    # Méthodes utilitaires XML

    def _get_text(self, elem, tag, default=''):
        child = elem.find(tag)
        return child.text.strip() if child is not None and child.text else default

    def _get_float(self, elem, tag, default=0.0):
        child = elem.find(tag)
        if child is not None and child.text:
            try:
                return float(child.text.strip())
            except ValueError:
                return default
        return default

    def _get_int(self, elem, tag, default=0):
        child = elem.find(tag)
        if child is not None and child.text:
            try:
                return int(child.text.strip())
            except ValueError:
                return default
        return default


# ------------------------------------------------------------------ #
# Test du module                                                        #
# ------------------------------------------------------------------ #

# if __name__ == "__main__":

#     print("="*55)
#     print("  TEST DU MODULE LOAD_MAP")
#     print("="*55)

#     for map_id in range(1, 9):
#         print(f"\n--- Map {map_id} ---")
#         m = Map()
#         try:
#             m.load_from_xml(f"../data/map{map_id}.xml")
#         except FileNotFoundError:
#             print(f"    Fichier map{map_id}.xml introuvable")

#     print("\n" + "="*55)
#     print("  Test termine")
#     print("="*55)
    