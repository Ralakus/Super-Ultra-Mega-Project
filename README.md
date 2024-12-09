# Super Ultra Mega Project
Repository for ENGE 1414 project.

## Getting Started
### Required VSCode Extensions
* Pyright
* Ruff
* Black Formatter
* autoDocstring

### Code Conventions
* All code MUST comply with PEP8 standards
* All code MUST be typed
    * If typing is not possible due to lack of types in libraries, subsitute the type with `typing.Any`
    * If the type exists but is inaccessible through a library, leave out type annotation and let pyright handle it
* McCabe complexity warnings SHOULD be followed
    * If ignored, they must be explicity ignored using a `#  noqa: ` entry 
* All code MUST be formatted using Black before commit
* All objects and functions SHOULD have descriptive descriptions
    * Sometimes descriptive names are enough, use your best judgement but make sure to fill out the docstrings
* Any name MUST NOT be abbreviated unless abbreviations are well known or defined in the project's glossary. For example:
    * Use `distance` instead of `dist`
    * Use `distribute` instead of `dist`
    * Use `distinguish` instead of `dist`
    * Use `distill` instead of `dist`
* Single variables are okay in mathmatical contexts but SHOULD be writen out of what that variable is describing

## Program
0. Read in map and traversal graph data
    * Map will be defined in JSON
    * Traversal graph will be defined in a JSON file
1. Generated n randomized instances of simple machine learning models
2. Get new position and rotation outputs from each model
3. If outputs don't comply with constraints, nudge each object to fit into the nearest valid constraint
4. Score each layout based on traversal graphs
5. Keep top scoring model and replace the rest of the models with variations of the top model based on an `alpha` value
6. Write top scoring output to file each iteration
7. Go to 2