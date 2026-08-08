"""
Guess a word using a simple geentic algorithm.
"""

from typing import NamedTuple

from ga_string.generation import DEFAULT_CROSSOVER_POINTS
from ga_string.generation import DEFAULT_CROSSOVER_RATE
from ga_string.generation import DEFAULT_MUTATION_RATE
from ga_string.generation import Amount
from ga_string.generation import Gene
from ga_string.generation import Generation
from ga_string.generation import Genes
from ga_string.generation import Individual
from ga_string.generation import MismatchedGeneCount
from ga_string.generation import Rate


def main() -> None:
    """
    Entry
    """
    args = _get_args()
    generation = Generation(
        args.population_size,
        target=args.target,
        crossover_rate=args.crossover_rate,
        crossover_points=args.crossover_points,
        mutation_rate=args.mutation_rate,
    )
    max_generations = int(
        input("Enter the max number of generations/iterations (int, > 0)\n: ")
    )
    generation_count = 1
    fittest: Individual = generation.fittest()
    while generation_count < max_generations:
        print(f"Generation #{generation_count}")
        print(f"Best guess {fittest}")
        generation = generation.next_generation()
        fittest = generation.fittest()
        generation_count += 1
        if fittest == args.target:
            break
    print(f"Generation #{generation_count}")
    print(f"Best guess: {fittest}")
    print(f"Fitness: {fittest.evaluate(args.target)}/{len(args.target)}")
    if generation_count == max_generations:
        print("Reached max number of generations")
    if fittest == args.target:
        print(
            f"Successfully guessed '{args.target}' in {generation_count} generations."
        )


class _Args(NamedTuple):
    """
    CLI args
    """

    target: Individual
    population_size: Amount
    crossover_rate: Rate
    crossover_points: Amount
    mutation_rate: Rate


def _get_args() -> _Args:
    """
    Get CLI args.
    """
    target_phrase = str(
        input(
            "Target individual/phrase to guess\n "
            f"(valid genes/characters: '{Gene.SPACE}'): "
        )
    )
    target = Individual(Genes.from_str(target_phrase))
    population_size = Amount(
        int(input("Individuals/guesses per generation (int, > 0)\n: "))
    )
    crossover_rate = Rate(
        float(
            input(f"Crossover rate (float [0,1], default={DEFAULT_CROSSOVER_RATE})\n: ")
            or DEFAULT_CROSSOVER_RATE
        )
    )
    crossover_points = Amount(
        int(
            input(
                "Crossover points (int [0, length of target], "
                f"default = {DEFAULT_CROSSOVER_POINTS})\n: "
            )
            or DEFAULT_CROSSOVER_POINTS
        )
    )
    if crossover_points >= len(target):
        raise MismatchedGeneCount(
            f"crossover points must be in [0, length of target], {crossover_points} was given"
        )
    mutation_rate = Rate(
        float(
            input(f"Mutation rate (float [0,1], default = {DEFAULT_MUTATION_RATE})\n: ")
            or DEFAULT_MUTATION_RATE
        )
    )
    return _Args(
        target=target,
        population_size=population_size,
        crossover_rate=crossover_rate,
        crossover_points=crossover_points,
        mutation_rate=mutation_rate,
    )
