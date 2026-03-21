import pytest
from src.chemistry.parser import ChemicalParser, ReactionEngine


class TestChemicalParser:
    def test_parse_simple_formula(self):
        assert ChemicalParser.parse_formula("Cu") == {"Cu": 1}

    def test_parse_formula_with_count(self):
        assert ChemicalParser.parse_formula("H2O") == {"H": 2, "O": 1}

    def test_parse_complex_formula(self):
        assert ChemicalParser.parse_formula("Al2O3") == {"Al": 2, "O": 3}

    def test_parse_formula_strips_spaces(self):
        result = ChemicalParser.parse_formula("Cu Zn")
        assert result == {"Cu": 1, "Zn": 1}

    def test_format_formula_basic(self):
        result = ChemicalParser.format_formula({"H": 2, "O": 1})
        assert result == "H2O"

    def test_format_formula_no_html(self):
        result = ChemicalParser.format_formula({"Al": 2, "O": 3})
        assert "<sub>" not in result
        assert result == "Al2O3"

    def test_format_empty(self):
        assert ChemicalParser.format_formula({}) == ""

    def test_format_single_element(self):
        assert ChemicalParser.format_formula({"Cu": 1}) == "Cu"


class TestReactionEngine:
    def test_add_molecule(self):
        engine = ReactionEngine()
        engine.add_molecule("Cu", 1)
        assert len(engine.molecules) == 1

    def test_add_direct_composition(self):
        engine = ReactionEngine()
        engine.add_direct_composition({"Fe": 70.0, "Cr": 18.0, "Ni": 12.0}, "Stainless")
        assert len(engine.molecules) == 1

    def test_total_composition_single_element(self):
        engine = ReactionEngine()
        engine.add_direct_composition({"Cu": 100.0}, "Pure Copper")
        comp = engine.get_total_composition()
        assert comp == {"Cu": 100.0}

    def test_total_composition_direct_returns_as_is(self):
        engine = ReactionEngine()
        engine.add_direct_composition({"Fe": 70.0, "C": 30.0}, "Steel")
        comp = engine.get_total_composition()
        assert abs(comp["Fe"] - 70.0) < 0.01
        assert abs(comp["C"] - 30.0) < 0.01

    def test_reaction_equation_display(self):
        engine = ReactionEngine()
        engine.add_molecule("Cu", 2)
        eq = engine.get_reaction_equation()
        assert "Cu" in eq

    def test_reaction_equation_with_coefficient(self):
        engine = ReactionEngine()
        engine.add_molecule("Cu", 3)
        eq = engine.get_reaction_equation()
        assert "3" in eq

    def test_clear(self):
        engine = ReactionEngine()
        engine.add_molecule("Cu", 1)
        engine.clear()
        assert len(engine.molecules) == 0

    def test_empty_equation(self):
        engine = ReactionEngine()
        assert engine.get_reaction_equation() == "No reaction defined"

    def test_empty_composition(self):
        engine = ReactionEngine()
        assert engine.get_total_composition() == {}
