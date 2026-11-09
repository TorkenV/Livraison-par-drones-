import xml.etree.ElementTree as ET
import os


class Drone:

    def __init__(self):
        # Initialisation du drone par défaut 
        self.name             = "VTOL" # Nom du drone
        self.description      = ""           # Description du drone (Fonction + Tache)

        self.cruise_speed     = 10.0    # Vitesse de croisière (m/s)
        self.max_speed        = 15.0    # Vitesse maximale (m/s)

        self.max_acceleration = 2.0     # Accélération maximale (m/s²)
        self.max_deceleration = 2.5     # Décélération maximale (m/s²)

        self.mass_empty       = 5.0     # Masse à vide (kg)
        self.mass_loaded      = 6.0     # Masse en charge (kg)

        self.altitude_min     = 20.0    # Altitude minimale (m)
        self.altitude_max     = 120.0   # Altitude maximale (m)

    def load_from_xml(self, filepath):
        # Verification de l'existence du fichier 
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier drone introuvable : {filepath}")
        
        # Chargement des caractéristiques du drone à partir du fichier XML 
        tree = ET.parse(filepath)
        root = tree.getroot()

        self.name             = self._get_str  (root, "name",                self.name)
        self.description      = self._get_str  (root, "description",      self.description)

        self.cruise_speed     = self._get_float(root, "cruise_speed",     self.cruise_speed)
        self.max_speed        = self._get_float(root, "max_speed",        self.max_speed)

        self.max_acceleration = self._get_float(root, "max_acceleration", self.max_acceleration)
        self.max_deceleration = self._get_float(root, "max_deceleration", self.max_deceleration)

        self.mass_empty       = self._get_float(root, "mass_empty",       self.mass_empty)
        self.mass_loaded      = self._get_float(root, "mass_loaded",      self.mass_loaded)

        self.altitude_min     = self._get_float(root, "altitude_min",     self.altitude_min)
        self.altitude_max     = self._get_float(root, "altitude_max",     self.altitude_max)

        self._print_summary(filepath)
        return self


    def _get_float(self, parent, tag, default):
        # Lit un tag XML et retourne un float (ou default si absent)
        elem = parent.find(tag)
        if elem is not None and elem.text:
            return float(elem.text)
        return default

    def _get_str(self, parent, tag, default):
        # Lit un tag XML et retourne une str (ou default si absent)
        elem = parent.find(tag)
        if elem is not None and elem.text:
            return elem.text
        return default

    def _print_summary(self, filepath):
        # Affiche un résumé après chargement 
        print("="*55)
        print(f"  Drone chargé : {self.name}")
        print(f"  Fichier      : {filepath}")
        print("="*55)
        print(f"  Vitesse croisière  : {self.cruise_speed} m/s")
        print(f"  Vitesse max        : {self.max_speed} m/s")
        print(f"  Accélération max   : {self.max_acceleration} m/s²")
        print(f"  Décélération max   : {self.max_deceleration} m/s²")
        print(f"  Masse à vide       : {self.mass_empty} kg")
        print(f"  Masse en charge    : {self.mass_loaded} kg")
        print(f"  Altitude min/max   : {self.altitude_min} / {self.altitude_max} m")
        print("="*55)

    def __str__(self):
        return (f"Drone('{self.name}', "
                f"cruise={self.cruise_speed} m/s, "
                f"mass={self.mass_loaded} kg)")



# Premier test pour recharger drone1
# if __name__ == "__main__":
#     d = Drone()
#     d.load_from_xml(r"C:\Users\Dr console\Desktop\INSA\4A\PIR\Newest work\data\drone1.xml")
#     print(d)
