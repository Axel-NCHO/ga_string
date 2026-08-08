"""
Guess a word using a simple geentic algorithm.
"""

from ga_string.generation import Gene
from ga_string.generation import Generation
from ga_string.generation import Genes
from ga_string.generation import Individual


def main() -> None:
    """
    Entry
    """
    target_phrase = str(
        input(
            "Enter an individual/phrase to guess\n "
            f"(valid genes/characters: '{Gene.SPACE}'): "
        )
    )
    target = Individual(Genes.from_str(target_phrase))
    population_size = int(
        input("Enter the number of individuals/guesses per generation (int, > 0)\n: ")
    )
    generation = Generation(population_size, target=target)
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
        if fittest == target:
            break
    print(f"Generation #{generation_count}")
    print(f"Best guess: {fittest}")
    print(f"Fitness: {fittest.evaluate(target)}/{len(target)}")
    if generation_count == max_generations:
        print("Reached max number of generations")
    if fittest == target:
        print(f"Successfully guessed '{target}' in {generation_count} generations.")
