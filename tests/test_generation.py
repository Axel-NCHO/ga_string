"""
Tests for module ga_string.generation
"""

from unittest.mock import patch

import pytest

from ga_string.generation import Generation
from ga_string.generation import Individual
from ga_string.generation import InvalidGene
from ga_string.generation import MismatchedGeneCount
from ga_string.generation import _MutableIndividual

# pylint: disable=protected-access


class TestIndividual:
    """
    Tests for ga_string.generation.Individual
    """

    def test_str(self) -> None:
        """
        Test string representation of an individual
        """
        individual = Individual("individual")
        assert str(individual) == "individual"

    def test_len(self) -> None:
        """
        Test the length of individual
        """
        individual = Individual("individual")
        assert len(individual) == 10

    def test_invalid_genes(self) -> None:
        """
        Test invalid gene
        """
        with pytest.raises(InvalidGene) as error:
            _ = Individual("individual?")
            assert error.value.gene == "?"

    def test_evaluate(self) -> None:
        """
        Test evaluate an individual against another one
        """
        individual = Individual("inxivixual")
        target = Individual("individual")
        assert individual.evaluate(target) == 8

    def test_evaluate_mismatched_gene_count(self) -> None:
        """
        Test evaluate an individual against another one
        """
        with pytest.raises(MismatchedGeneCount):
            individual = Individual("abcd")
            target = Individual("individual")
            _ = individual.evaluate(target)


# pylint: disable=too-few-public-methods
class TestMutableIndividual:
    """
    Tests for ga_string.generation._MutableIndividual
    """

    def test_from_value(self) -> None:
        """
        Test creating a mutable individual from a value
        """
        mut_individual = _MutableIndividual.from_value("individual")
        assert str(mut_individual) == "individual"


# pylint: enable=too-few-public-methods


class TestGeneration:
    """
    Tests for ga_string.generation.Generation
    """

    def test_len(self) -> None:
        """
        Test getting the size of a generation
        """
        generation = Generation(10, target=Individual("individual"))
        assert len(generation) == 10

    def test_make_children(self) -> None:
        """
        Test make two children from two parents
        """
        i1 = Individual("aaaabbbb")
        i2 = Individual("ccccdddd")
        with patch(
            "ga_string.generation.Individual.random_crossover_point", return_value=2
        ):
            c1, c2 = Generation._make_children(i1, i2)
            assert str(c1) == "aaccdddd"
            assert str(c2) == "ccaabbbb"

    def test_make_children_mismatched_gene_count(self) -> None:
        """
        Test make two children from two parents with different gene count
        """
        i1 = Individual("aaaabbbb")
        i2 = Individual("ccccdddde")
        with pytest.raises(MismatchedGeneCount):
            _ = Generation._make_children(i1, i2)

    def test_fittests(self) -> None:
        """
        Test the top 50% fittest individuals of a generation
        """
        generation = Generation(2, target=Individual("individual"))
        generation._individuals = [
            _MutableIndividual.from_value("individuax"),
            _MutableIndividual.from_value("individuxx"),
        ]
        assert generation._fittests() == [_MutableIndividual.from_value("individuax")]
        assert generation.fittest() == Individual("individuax")
