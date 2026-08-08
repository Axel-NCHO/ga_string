"""
A generation is made of several individuals/words. Each individual
is a guess.

A new generation can be created from a previous one.
To produce a new generation (`next generation`), new individuals are created
from the fittest ones in the previous generation. Then these new individuals
are mutated randomly with a small mutation rate, before being added to the new
generation. To avoid losing the genes of the fittest individual of the previous
generation, the algorithm implements elitism by copying the fittest individual
for the previous generation into the next one. The new generation has the same
number of individuals as the previous one.

The fitness of an individual is the number of its genes/characters that match
the target individual.
"""

import itertools
import random
from collections.abc import Iterable
from copy import deepcopy
from typing import Final
from typing import NewType
from typing import Self
from typing import overload


class InvalidGene(Exception):
    """
    Raised when trying to create an individual with at least one invalid gene.
    """

    def __init__(self, gene: str) -> None:
        self.gene: Final[str] = gene
        super().__init__(f"invalid gene: {gene}")


class MismatchedGeneCount(Exception):
    """
    Raised when trying to perform an operation involving two genes that requires
    them to have the same number of genes.
    """


class EmptyGeneration(Exception):
    """
    Raised when creating a generation of negative of null size.
    """


class Generation:
    """
    A generation is made of several individuals/guesses.
    """

    def __init__(self, size: int, *, target: Individual) -> None:
        """
        Create a new generation.

        Raises:
            EmptyGeneration
        """
        if size <= 0:
            raise EmptyGeneration
        self._target = target
        self._individuals = [_MutableIndividual(len(target)) for _ in range(size)]

    def __len__(self) -> int:
        return len(self._individuals)

    def crossover(self) -> Generation:
        """
        Create a new generation from this generation by selecting the fittest
        individuals and creating offsprings from them.
        The new generation only contains the offsprings and has the same size
        as this geenration.
        """
        parents = self._fittests()
        children: list[_MutableIndividual] = [parents[0].copy()]  # elitism
        while len(children) != len(self):
            parents = random.choices(parents, k=2)
            child1, child2 = self._make_children(parents[0], parents[1])
            # add children one at a time to avoid going over the population size
            for child in child1, child2:
                if len(children) != len(self):
                    child.mutate()
                    children.append(child)
        generation = Generation(len(self), target=self._target)
        generation._individuals = children  # pylint: disable=protected-access
        return generation

    def _fittests(self) -> list[_MutableIndividual]:
        """
        Return the top 50% fittest individuals sorted in decreasing value of fitness.
        """
        all_sorted = sorted(
            self._individuals, key=lambda ind: ind.evaluate(self._target), reverse=True
        )
        return all_sorted[: len(self._individuals) // 2]

    def fittest(self) -> Individual:
        """
        Return the fittest individual of this generation.
        """
        return self._fittests()[0]

    @classmethod
    def _make_children(
        cls, i1: Individual, i2: Individual
    ) -> tuple[_MutableIndividual, _MutableIndividual]:
        """
        Create two new individuals from two parents with the same gene count.
        First choose a random point in the gene sequance of the parents.
        Then each child is comprised of the concatenation of the genes on either side
        of the choosen point on each parent.

        Raises:
            MismatchedGeneCount
        """
        if len(i1) != len(i2):
            raise MismatchedGeneCount("make children")
        nb_genes = len(i1)
        crossover_point = i1.random_crossover_point()
        genes1 = itertools.chain(i1[0:crossover_point], i2[crossover_point:nb_genes])
        genes2 = itertools.chain(i2[0:crossover_point], i1[crossover_point:nb_genes])
        return _MutableIndividual.from_value(genes1), _MutableIndividual.from_value(
            genes2
        )


class Individual:
    """
    An individual
    """

    GENE_SPACE: Final[str] = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, value: str) -> None:
        """
        Create a new individual

        Raises:
            InvalidGene
        """
        for gene in value:
            if gene not in self.GENE_SPACE:
                raise InvalidGene(gene)
        self._genes = list(value)

    def __len__(self) -> int:
        return len(self._genes)

    def __eq__(self, value: object, /) -> bool:
        if isinstance(value, Individual):
            return self._genes == value._genes
        return NotImplemented

    def __str__(self) -> str:
        return "".join(self._genes)

    def __repr__(self) -> str:
        return f"Individual({self.__str__()})"

    @overload
    def __getitem__(self, idx: int) -> str: ...
    @overload
    def __getitem__(self, idx: slice) -> list[str]: ...
    def __getitem__(self, idx: int | slice) -> str | list[str]:
        return self._genes[idx]

    def random_crossover_point(self) -> int:
        """
        Random index between 1 and len - 1, both iuncluded
        """
        return random.randint(1, len(self) - 1)

    def evaluate(self, target: Individual, /) -> _Fitness:
        """
        Evaluate this individual against the target.

        Raises:
            MismatchedGeneCount
        """
        if len(self) != len(target):
            raise MismatchedGeneCount("evaluate")
        raw_fitness: int = 0
        for i in range(len(self)):
            if self._genes[i] == target._genes[i]:  # pylint: disable=protected-access
                raw_fitness += 1
        return _Fitness(raw_fitness)

    def copy(self) -> Self:
        """
        Copy this individual
        """
        return deepcopy(self)


class _MutableIndividual(Individual):
    """
    Individual whose genes can be mutated.
    Such individual is always initialized with random genes.
    """

    MUTATION_RATE: Final[float] = 0.05
    """
    Probability of mutating a gene
    """

    def __init__(self, nb_genes: int, /) -> None:
        super().__init__(self._random_value(nb_genes))

    @classmethod
    def from_value(cls, value: Iterable[str], /) -> Self:
        """
        Create a mutable individual from a value
        """
        all_genes = list(value)
        this = cls(len(all_genes))
        this._genes = all_genes
        return this

    def mutate(self) -> None:
        """
        Replace the genes of this individual by random genes with a probability of `mutation_pb`
        """
        for i in range(len(self)):
            if random.random() < self.MUTATION_RATE:
                self._genes[i] = self._random_gene()

    @classmethod
    def _random_value(cls, lenght: int, /) -> str:
        return "".join(random.choices(cls.GENE_SPACE, k=lenght))

    @classmethod
    def _random_gene(cls) -> str:
        return random.choice(cls.GENE_SPACE)


_Fitness = NewType("_Fitness", int)
