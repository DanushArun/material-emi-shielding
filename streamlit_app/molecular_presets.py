"""
Molecular Presets for Chemical EMI Designer
Common molecules and compounds for quick selection
"""

MOLECULAR_PRESETS = {
    "Water": {
        "formula": "H2O",
        "composition": {"H": 2, "O": 1},
        "description": "Water molecule",
        "category": "Common"
    },
    "Carbon Dioxide": {
        "formula": "CO2",
        "composition": {"C": 1, "O": 2},
        "description": "Carbon dioxide",
        "category": "Common"
    },
    "Iron Oxide": {
        "formula": "Fe2O3",
        "composition": {"Fe": 2, "O": 3},
        "description": "Rust, ferric oxide",
        "category": "Metal Oxide"
    },
    "Aluminum Oxide": {
        "formula": "Al2O3",
        "composition": {"Al": 2, "O": 3},
        "description": "Alumina, corundum",
        "category": "Metal Oxide"
    },
    "Copper Oxide": {
        "formula": "CuO",
        "composition": {"Cu": 1, "O": 1},
        "description": "Cupric oxide",
        "category": "Metal Oxide"
    },
    "Silicon Dioxide": {
        "formula": "SiO2",
        "composition": {"Si": 1, "O": 2},
        "description": "Silica, quartz",
        "category": "Metal Oxide"
    },
    "Titanium Dioxide": {
        "formula": "TiO2",
        "composition": {"Ti": 1, "O": 2},
        "description": "Titania",
        "category": "Metal Oxide"
    },
    "Graphene": {
        "formula": "C6",
        "composition": {"C": 6},
        "description": "Hexagonal carbon lattice",
        "category": "Carbon"
    },
    "Diamond": {
        "formula": "C8",
        "composition": {"C": 8},
        "description": "Cubic carbon lattice",
        "category": "Carbon"
    },
    "Carbon Nanotube": {
        "formula": "C20",
        "composition": {"C": 20},
        "description": "Cylindrical carbon structure",
        "category": "Carbon"
    },
    "Ferrite": {
        "formula": "Fe3O4",
        "composition": {"Fe": 3, "O": 4},
        "description": "Magnetite, magnetic iron oxide",
        "category": "Magnetic"
    },
    "Nickel Ferrite": {
        "formula": "NiFe2O4",
        "composition": {"Ni": 1, "Fe": 2, "O": 4},
        "description": "Magnetic ceramic",
        "category": "Magnetic"
    },
    "Cobalt Ferrite": {
        "formula": "CoFe2O4",
        "composition": {"Co": 1, "Fe": 2, "O": 4},
        "description": "Hard magnetic material",
        "category": "Magnetic"
    },
    "Silver Nanowire": {
        "formula": "Ag12",
        "composition": {"Ag": 12},
        "description": "High conductivity nanowire",
        "category": "Conductor"
    },
    "Copper Sulfide": {
        "formula": "Cu2S",
        "composition": {"Cu": 2, "S": 1},
        "description": "Semiconductor material",
        "category": "Semiconductor"
    },
    "Zinc Oxide": {
        "formula": "ZnO",
        "composition": {"Zn": 1, "O": 1},
        "description": "Wide bandgap semiconductor",
        "category": "Semiconductor"
    },
    "Indium Tin Oxide": {
        "formula": "In2SnO5",
        "composition": {"In": 2, "Sn": 1, "O": 5},
        "description": "Transparent conductor",
        "category": "Conductor"
    },
    "Molybdenum Disulfide": {
        "formula": "MoS2",
        "composition": {"Mo": 1, "S": 2},
        "description": "2D semiconductor",
        "category": "2D Material"
    },
    "Tungsten Disulfide": {
        "formula": "WS2",
        "composition": {"W": 1, "S": 2},
        "description": "Transition metal dichalcogenide",
        "category": "2D Material"
    },
    "Boron Nitride": {
        "formula": "BN",
        "composition": {"B": 1, "N": 1},
        "description": "White graphene",
        "category": "2D Material"
    },
    "Metal-Organic Framework": {
        "formula": "C24H12Cu3O12",
        "composition": {"C": 24, "H": 12, "Cu": 3, "O": 12},
        "description": "Porous crystalline material",
        "category": "Advanced"
    },
    "Perovskite": {
        "formula": "CaTiO3",
        "composition": {"Ca": 1, "Ti": 1, "O": 3},
        "description": "Crystal structure type",
        "category": "Advanced"
    },
    "Gallium Arsenide": {
        "formula": "GaAs",
        "composition": {"Ga": 1, "As": 1},
        "description": "III-V semiconductor",
        "category": "Semiconductor"
    },
    "Silicon Carbide": {
        "formula": "SiC",
        "composition": {"Si": 1, "C": 1},
        "description": "Wide bandgap semiconductor",
        "category": "Semiconductor"
    },
    "Gallium Nitride": {
        "formula": "GaN",
        "composition": {"Ga": 1, "N": 1},
        "description": "Wide bandgap semiconductor",
        "category": "Semiconductor"
    }
}

REACTION_PRESETS = {
    "Steel Formation": {
        "molecules": [
            {"formula": "Fe2O3", "coefficient": 1},
            {"formula": "C3", "coefficient": 3},
            {"formula": "Al", "coefficient": 2}
        ],
        "description": "Steel alloy formation reaction"
    },
    "Ferrite Composite": {
        "molecules": [
            {"formula": "Fe3O4", "coefficient": 2},
            {"formula": "C20", "coefficient": 1},
            {"formula": "Cu", "coefficient": 4}
        ],
        "description": "Magnetic ferrite composite"
    },
    "Conductive Polymer": {
        "molecules": [
            {"formula": "C6H4", "coefficient": 10},
            {"formula": "Ag12", "coefficient": 1},
            {"formula": "Cu2S", "coefficient": 2}
        ],
        "description": "Conductive polymer composite"
    },
    "High-Performance Shield": {
        "molecules": [
            {"formula": "C20", "coefficient": 1},
            {"formula": "Ag12", "coefficient": 2},
            {"formula": "Fe3O4", "coefficient": 3},
            {"formula": "MoS2", "coefficient": 1}
        ],
        "description": "Advanced multi-material shield"
    },
    "Metamaterial": {
        "molecules": [
            {"formula": "Cu", "coefficient": 8},
            {"formula": "SiO2", "coefficient": 4},
            {"formula": "Au", "coefficient": 2},
            {"formula": "C6", "coefficient": 6}
        ],
        "description": "Electromagnetic metamaterial"
    }
}