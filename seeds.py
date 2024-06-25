import json
import random

# List of base material types
base_material_types = [
    "Wood", "Metal", "Plastic", "Glass", "Ceramic", 
    "Concrete", "Textile", "Leather", "Rubber", "Stone"
]

# Function to generate random details
def generate_details(material):
    # Base details for all materials
    base_details = {
        "density": round(random.uniform(0.5, 10.0), 2),
        "hardness": round(random.uniform(1.0, 10.0), 2),
        "color": random.choice(["Red", "Blue", "Green", "Yellow", "Black", "White", "Gray", "Brown"]),
        "texture": random.choice(["Smooth", "Rough", "Grainy", "Matte", "Glossy"]),
    }
    
    # Additional details for specific materials
    if material.startswith("Wood") or material.startswith("Textile"):
        base_details["flammability"] = random.choice(["High", "Medium", "Low"])
    if material.startswith("Metal") or material.startswith("Plastic") or material.startswith("Glass"):
        base_details["conductivity"] = round(random.uniform(0.1, 100.0), 2)
    if material.startswith("Leather") or material.startswith("Rubber"):
        base_details["elasticity"] = round(random.uniform(0.1, 5.0), 2)
    
    return base_details

# Generate materials list
materials = []
for i in range(100):
    base_material = random.choice(base_material_types)
    material_name = f"{base_material}_{i+1}"
    material_details = generate_details(material_name)
    materials.append({
        "material": material_name,
        "details": material_details
    })

# Convert to JSON
materials_json = json.dumps(materials, indent=4)

# Save to a JSON file
with open("materials.json", "w") as file:
    file.write(materials_json)

print("Materials JSON generated and saved to materials.json")
