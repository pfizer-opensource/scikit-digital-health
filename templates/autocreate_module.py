"""
Automatically create a base module that can be modified

Lukas Adamowicz
2020, Pfizer DMTI
"""
from datetime import datetime
from pathlib import Path


def file_replace(lines, name, replace_with):
    for i, line in enumerate(lines):
        if '!{{ ' + name + ' }}' in line:
            lines[i] = line.replace('!{{ ' + name + ' }}', replace_with)

    return lines


init_contents = '''"""
MODULE DESCRIPTION (:mod:`PfyMU.{mod_lower}`)
=======================================

.. currentmodule:: PfyMU.{mod_lower}

Heading 1
---------

.. autosummary::
    :toctree: generated/

    {mod}
    
Heading 2
---------
contents
"""
from PfyMU.{mod_lower}.{mod_lower} import {mod}

'''


if __name__ == "__main__":
    module_name = input("Module name: ")
    author = input("Author: ")
    short_descr = input("Short description: ")
    year = datetime.now().year

    if not module_name[0].isupper():
        module_name = module_name.capitalize()

    if module_name == '' or author == '' or short_descr == '':
        raise ValueError('please provide module name, author, and a short description')

    with open('templates/core_template.txt', 'r') as f:
        lines = f.readlines()

    file_replace(lines, 'module_name', module_name)
    file_replace(lines, 'author', author)
    file_replace(lines, 'short_description', short_descr)
    file_replace(lines, 'year', year)

    folder = Path('src/PfyMU') / module_name.lower()

    folder.mkdir()

    core_file = folder / f'{module_name.lower()}.py'
    with open(core_file, 'w') as f:
        f.writelines(lines)

    init_file = folder / '__init__.py'
    with open(init_file, 'w') as f:
        f.write(
            init_contents.format(mod_lower=module_name.lower(), mod=module_name)
        )
