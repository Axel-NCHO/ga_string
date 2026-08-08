"""
Tests for module ga_string.generation
"""

from unittest.mock import patch

import pytest

from ga_string.generation import Amount
from ga_string.generation import Generation
from ga_string.generation import Genes
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
        individual = Individual(Genes.from_str("individual"))
        assert str(individual) == "individual"

    def test_repr(self) -> None:
        """
        Test that individual can be recreated with its repr.
        """
        individual = Individual(Genes.from_str("aaaabbbb"))
        twin = eval(repr(individual))  # pylint: disable=eval-used
        assert isinstance(twin, Individual)
        assert individual == twin

    def test_len(self) -> None:
        """
        Test the length of individual
        """
        individual = Individual(Genes.from_str("individual"))
        assert len(individual) == 10

    def test_invalid_genes(self) -> None:
        """
        Test invalid gene
        """
        with pytest.raises(InvalidGene) as error:
            _ = Individual(Genes.from_str("individual#"))
        assert error.value.gene == "#"

    def test_evaluate(self) -> None:
        """
        Test evaluate an individual against another one
        """
        individual = Individual(Genes.from_str("inxivixual"))
        target = Individual(Genes.from_str("individual"))
        assert individual.evaluate(target) == 8

    def test_evaluate_mismatched_gene_count(self) -> None:
        """
        Test evaluate an individual against another one
        """
        with pytest.raises(MismatchedGeneCount):
            individual = Individual(Genes.from_str("abcd"))
            target = Individual(Genes.from_str("individual"))
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
        mut_individual = _MutableIndividual._with_genes(Genes.from_str("individual"))
        assert str(mut_individual) == "individual"

    @pytest.mark.parametrize("nb_points", [Amount(i) for i in range(1, 8)])
    def test_random_crossover_points(self, nb_points: Amount) -> None:
        """
        Test generated random crossover points.
        """
        individual = _MutableIndividual._with_genes(Genes.from_str("aaaabbbb"))
        points = individual._random_crossover_points(nb_points)
        assert len(points) == nb_points
        for i in range(nb_points - 1):
            assert points[i] < points[i + 1]

    def test_random_crossover_points_exceeds(self) -> None:
        """
        Test that generating random crossover points fails if the amount exceeds
        the length of the infividual.
        """
        individual = _MutableIndividual._with_genes(Genes.from_str("aaaabbbb"))
        with pytest.raises(MismatchedGeneCount):
            _ = individual._random_crossover_points(Amount(10))

    def test_make_children(self) -> None:
        """
        Test make two children from two parents
        """
        i1 = _MutableIndividual._with_genes(Genes.from_str("aaaabbbb"))
        i2 = _MutableIndividual._with_genes(Genes.from_str("ccccdddd"))
        with patch(
            "ga_string.generation._MutableIndividual._random_crossover_points",
            return_value=[2, 5],
        ):
            c1, c2 = i1.make_children(i2, crossover_points=Amount(2))
            assert str(c1) == "aaccdbbb"
            assert str(c2) == "ccaabddd"

    def test_make_children_mismatched_gene_count(self) -> None:
        """
        Test make two children from two parents with different gene count
        """
        i1 = _MutableIndividual._with_genes(Genes.from_str("aaaabbbb"))
        i2 = _MutableIndividual._with_genes(Genes.from_str("ccccdddde"))
        with pytest.raises(MismatchedGeneCount):
            _ = i1.make_children(i2, crossover_points=Amount(1))


# pylint: enable=too-few-public-methods


class TestGeneration:
    """
    Tests for ga_string.generation.Generation
    """

    def test_len(self) -> None:
        """
        Test getting the size of a generation
        """
        generation = Generation(
            Amount(10), target=Individual(Genes.from_str("individual"))
        )
        assert len(generation) == 10

    def test_top_fifty_percent(self) -> None:
        """
        Test the top 50% fittest individuals of a generation (0.5 default crossover rate).
        """
        generation = Generation(
            Amount(2), target=Individual(Genes.from_str("individual"))
        )
        generation._individuals = [
            _MutableIndividual._with_genes(Genes.from_str("individuax")),
            _MutableIndividual._with_genes(Genes.from_str("individuxx")),
        ]
        assert generation._select_fittests() == [
            _MutableIndividual._with_genes(Genes.from_str("individuax"))
        ]
        assert generation.fittest() == Individual(Genes.from_str("individuax"))
