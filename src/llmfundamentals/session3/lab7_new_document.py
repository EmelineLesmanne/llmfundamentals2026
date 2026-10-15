"""
LAB 7 - A new document, on your own

Goal: do the whole job alone on a new document: a recipe card (data/recipe.txt).

What to do:
  1. Write the class Recipe with these 4 fields (names are imposed, types are yours to choose):
       name, servings, prep_minutes, ingredients
     Read the recipe: which value is a number? which one is a list? can one be missing?
  2. Call ask_with_retry on the recipe text and print the result.

Bonus: add a rule: servings must be at least 1. Test it with a fake answer you write yourself.

"""
from lab6_retry_loop import ask_with_retry
from lab_tools import DATA, get_llm
from loguru import logger
from pydantic import BaseModel


class Recipe(BaseModel):
    pass  # TODO: write the 4 fields


# Everything below only runs when you run this file directly.
if __name__ == "__main__":
    if len(Recipe.model_fields) < 4:
        logger.info("TODO not done yet: the class Recipe needs 4 fields")
    else:
        llm = get_llm()
        recipe_text = (DATA / "recipe.txt").read_text(encoding="utf-8")
        prompt = f"Extract the data of this recipe.\n\n{recipe_text}"
        recipe = ask_with_retry(llm, prompt, Recipe)
        logger.info(recipe)
