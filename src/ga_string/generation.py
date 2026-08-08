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
from collections.abc import Iterator
from copy import deepcopy
from typing import Final
from typing import NewType
from typing import Self
from typing import cast
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
    Raised when trying to perform an operation involving two individuals that
    requires them to have the same number of genes.
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

    def next_generation(self) -> Generation:
        """
        Create a new generation from this generation by selecting the fittest
        individuals and creating offsprings from them.
        The new generation only contains the offsprings and has the same size
        as this geenration.
        """
        parents = self._select_fittests()
        children = self._crossover(parents, mutate_children=True)
        next_gen = Generation(len(self), target=self._target)
        next_gen._individuals = children  # pylint: disable=protected-access
        return next_gen

    def _select_fittests(self) -> list[_MutableIndividual]:
        """
        Select the fittest individuals
        """
        return self._top_fifty_percent()

    def _top_fifty_percent(self) -> list[_MutableIndividual]:
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
        return self._select_fittests()[0]

    def _crossover(
        self, parents: list[_MutableIndividual], *, mutate_children: bool = True
    ) -> list[_MutableIndividual]:
        """
        Create children from the given parents.
        The result has the same size as this generation.
        If `mutate_children`, introduce random mutations to children after
        crossover.
        """
        children: list[_MutableIndividual] = [parents[0].copy()]  # elitism
        while len(children) != len(self):
            [parent1, parent2] = random.choices(parents, k=2)
            child1, child2 = parent1.make_children(parent2)
            # add children one at a time to avoid going over the population size
            for child in child1, child2:
                if len(children) == len(self):
                    break
                if mutate_children:
                    child.mutate()
                children.append(child)
        return children


class Individual:
    """
    An individual
    """

    def __init__(self, value: Genes) -> None:
        """
        Create a new individual

        Raises:
            InvalidGene
        """
        self._genes: Genes = value

    def __len__(self) -> int:
        return len(self._genes)

    def __eq__(self, value: object, /) -> bool:
        if isinstance(value, Individual):
            return self._genes == value._genes
        return NotImplemented

    def __str__(self) -> str:
        return "".join(self._genes)

    def __repr__(self) -> str:
        return f"Individual(Genes.from_str('{self.__str__()}'))"

    @overload
    def __getitem__(self, idx: int) -> Gene: ...
    @overload
    def __getitem__(self, idx: slice) -> Genes: ...
    def __getitem__(self, idx: int | slice) -> Gene | Genes:
        if isinstance(res := self._genes[idx], Gene):
            return res
        return Genes._from_list(res)

    def __iter__(self) -> Iterator[Gene]:
        return iter(self._genes)

    def evaluate(self, target: Individual, /) -> _Fitness:
        """
        Evaluate this individual against the target.

        Raises:
            MismatchedGeneCount
        """
        if len(self) != len(target):
            raise MismatchedGeneCount("evaluate")
        raw_fitness: int = sum(
            gene == target_gene for gene, target_gene in zip(self, target)
        )
        return _Fitness(raw_fitness)

    def copy(self) -> Self:
        """
        Copy this individual
        """
        return deepcopy(self)


class Gene(str):
    """
    A gene.
    """

    SPACE: Final[str] = "abcdefghijklmnopqrstuvwxyz_,!:.-? ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    """
    All possible values of a gene
    """

    def __new__(cls, value: str) -> Self:
        """
        Create a new gene from a string value.

        Raises:
            InvalidGene:
        """
        if len(value) != 1:
            raise InvalidGene(value)
        if value not in cls.SPACE:
            raise InvalidGene(value)
        return super().__new__(cls, value)

    @classmethod
    def random(cls) -> Self:
        """
        Returns a random gene.
        """
        return cls(random.choice(cls.SPACE))

    @classmethod
    def random_k(cls, k: int = 1, /) -> list[Self]:
        """
        Returns `k` random genes.
        """
        if k <= 0:
            raise ValueError("generate no genes")
        return [cls(c) for c in random.choices(cls.SPACE, k=k)]


class _UncheckedGene(Gene):
    """
    A gene created without checking that the `str` is a valid gene.
    The given string MUST be a valid gene in the gene space.
    """

    def __new__(cls, value: str) -> Self:
        """
        Create a new gene from a string value without checking that
        the string is a valid gene.
        The given string MUST be a valid gene in the gene space.
        """
        return str.__new__(cls, value)


class Genes(list[Gene]):
    """
    A sequence of `Gene`s.
    """

    @classmethod
    def from_str(cls, value: str) -> Self:
        """
        Returns a sequence of genes from a string
        """
        return cls(Gene(c) for c in value)

    @classmethod
    def random(cls, length: int, /) -> Self:
        """
        Returns random genes.
        """
        return cls._from_list(_UncheckedGene.random_k(length))

    @classmethod
    def _from_str_unchecked(cls, value: str) -> Self:
        """
        Returns a sequence of genes from without checking if characters
        are valid genes. Characters MUST be valid genes.
        """
        return cls(_UncheckedGene(c) for c in value)

    @classmethod
    def _from_list(
        cls, value: list[Gene | _UncheckedGene] | list[Gene] | list[_UncheckedGene]
    ) -> Self:
        """
        Casts the input. A list of genes is already a valid `Genes`.
        """
        return cast(Self, value)

    @classmethod
    def from_iterable(cls, value: Iterable[Gene]) -> Self:
        """
        Collects the input in a list and casts it.
        """
        return cls._from_list(list(value))


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
        super().__init__(Genes.random(nb_genes))

    @classmethod
    def _with_genes(cls, value: Genes, /) -> Self:
        """
        Create a mutable individual from a value
        """
        this = cls(len(value))
        this._genes = value
        return this

    def mutate(self) -> None:
        """
        Replace the genes of this individual by random genes with a probability of `mutation_pb`
        """
        for i in range(len(self)):
            if random.random() < self.MUTATION_RATE:
                self._genes[i] = _UncheckedGene.random()

    def make_children(self, coparent: Self) -> tuple[Self, Self]:
        """
        Create two new individuals from this individual and a coparent with the same gene count.
        First choose a random point in the gene sequance of the parents.
        Then each child is comprised of the concatenation of the genes on either side
        of the choosen point on each parent.

        Raises:
            MismatchedGeneCount
        """
        if len(self) != len(coparent):
            raise MismatchedGeneCount("make children")
        nb_genes = len(self)
        crossover_point = self._random_crossover_point()
        genes1 = itertools.chain(
            self[0:crossover_point], coparent[crossover_point:nb_genes]
        )
        genes2 = itertools.chain(
            coparent[0:crossover_point], self[crossover_point:nb_genes]
        )
        return self._with_genes(Genes.from_iterable(genes1)), self._with_genes(
            Genes.from_iterable(genes2)
        )

    def _random_crossover_point(self) -> int:
        """
        Random index between 1 and len - 1, both iuncluded
        """
        return random.randint(1, len(self) - 1)


_Fitness = NewType("_Fitness", int)
