import io
import base64
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def validate_smiles(smiles):
    """
    Validate the SMILES string using RDKit. 
        
        Args:
            smiles: SMILES string to validate
        Returns:
            tuple: (is_valid, molecule) where is_valid is a boolean and molecule is the RDKit molecule object if valid, or None
    """

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol is not None, mol
    except Exception as e:
        logger.error(f"Error validating SMILES {smiles}: {str(e)}")
        return False, None

def get_molecule_image(mol, size=(400,300), highlight_atoms=None, highlight_bonds=None):
    """
    Generate an image of a molecule.

    Args:
        mol: RDKit molecule object
        size: Image size as (width, height) tuple
        highlight_atoms: List of atom indices to highlight
        highlight_bonds: List of bond indices to highlight

    Returns:
        PIL Image object of the molecule
    """
    try:
        # prep the molecule for drawing
        mol = Chem.Mol(mol)
        mol.RemoveAllConformers()
        AllChem.Compute2DCoords(mol)

        # draw the molecule
        drawer = Draw.MolDraw2DCairo(*size)
        if highlight_atoms:
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms if highlight_atoms else [], highlightBonds = highlight_bonds if highlight_bonds else [])
        else:
            drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        # convert to PIL image
        png_data = drawer.GetDrawingText()
        return Image.open(io.BytesIO(png_data))
    except Exception as e:
        logger.error(f"Error generating molecule image: {str(e)}")
        raise

def smiles_to_base64_image(smiles, size=(400,300)):
    """
    Convert a SMILES string to a base64 encoded image.

    Args:
        smiles: SMILES string to convert
        size: Image size as (width, height) tuple
        
    Returns:
        Base64 encoded image string of None on conversion failure
    """
    try:
        # validate the smiles
        is_valid, mol = validate_smiles(smiles)
        if not is_valid or mol is None:
            logger.warning(f"Invalid SMILES string: {smiles}")
            return None
        
        # generate img
        img = get_molecule_image(mol, size)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_base64
    except Exception as e:
        logger.error(f"Error converting SMILES to base64 image: {str(e)}")
        return None
        
def get_sample_molecules():
    """
    Get a list of sample molecules with varying solubility levels.
    
    Returns:
        List of dictionaries containing molecule information
    """
    return [
        {"name": "Glucose", "smiles": "C(C1C(C(C(C(O1)O)O)O)O)O", "solubility_level": "Very High Solubility"},
        {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "solubility_level": "Moderate-High Solubility"},
        {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "solubility_level": "Moderate Solubility"},
        {"name": "Paracetamol", "smiles": "CC(=O)NC1=CC=C(C=C1)O", "solubility_level": "Moderate Solubility"},
        {"name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "solubility_level": "Low Solubility"},
        {"name": "Cholesterol", "smiles": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C", "solubility_level": "Very Low Solubility"}
    ]
