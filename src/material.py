import json

class Details:
    def __init__(self, density, hardness, color, texture, flammability=None, conductivity=None, elasticity=None):
        self.details = [density, hardness, color, texture, flammability, conductivity, elasticity]
    
    def __str__(self):
        string = f"Density: {self.details[0]}, Hardness: {self.details[1]}, Color: {self.details[2]}, Texture: {self.details[3]}"
        if self.details[4]:
            string += f", Flammability: {self.details[4]}"
        if self.details[5]:
            string += f", Conductivity: {self.details[5]}"
        if self.details[6]:
            string += f", Elasticity: {self.details[6]}"
        
        return string
    
class Material:

    def __init__(self, material : str, details : Details):
        self.material = material
        self.details = details
    
    def __str__(self):
        return f"{self.material}: {self.details}"
    
    def to_dict(self):
        return {
            "material": self.material,
            "details": {
                "density": self.details.details[0],
                "hardness": self.details.details[1],
                "color": self.details.details[2],
                "texture": self.details.details[3],
                "flammability": self.details.details[4],
                "conductivity": self.details.details[5],
                "elasticity": self.details.details[6]
            }
        }
        
def read_materials_from_json(file_path):
    """Reads materials data from a JSON file and gives back a dictionary of materials-details."""
    with open(file_path, 'r') as file:
        materials_data = json.load(file)
    materials = []
    for material in materials_data:
        details = Details(**material["details"])
        materials.append(Material(material["material"], details))
    return materials


    

    
    