# -*- coding: utf-8 -*-
"""Automatically create *.rst files for sphinx for all doxygen+breathe created C++ api docs."""
from os import listdir
from os.path import isfile, join, splitext


CPP_FILE_PATH = join(
        'moe',
        'optimal_learning',
        'cpp',
        )


def get_cpp_files():
    """Find all files in CPP_FILE_PATH that are *.hpp or *.cpp."""
    cpp_files = {}
    for file_name in listdir(CPP_FILE_PATH):
        if isfile(join(CPP_FILE_PATH, file_name)):
            file_base, file_ext = splitext(file_name)
            # Skip any files that don't have an extension
            if not file_ext:
                continue

            if file_base not in cpp_files and file_ext in ['.hpp', '.cpp']:
                cpp_files[file_base] = {
                        'cpp': None,
                        'hpp': None,
                        }
            if file_ext == '.hpp':
                cpp_files[file_base]['hpp'] = file_name
            elif file_ext == '.cpp':
                cpp_files[file_base]['cpp'] = file_name
    return cpp_files


def create_cpp_tree(cpp_files):
    """Create cpp_tree.rst index file."""
    fout = open(
            join(
                'docs',
                'cpp_tree.rst',
                ),
            'w'
    )
    fout.write("""
C++ Files
=========

.. toctree::
    :maxdepth: 2

"""
    )
    for cpp_file in cpp_files:
        fout.write('    {0:s}.rst\n'.format(cpp_file))
    fout.close()


def create_rst_file(file_base, files):
    """Create a default file_base.rst file for sphinx."""
    fout = open(
            join(
                'docs',
                '{0:s}.rst'.format(file_base),
                ),
            'w'
    )
    fout.write(
"""{0:s}
=====

**Contents:**

""".format(file_base)
    )
    index_count = 1
    for file_type in ['hpp', 'cpp']:
        if files[file_type] is not None:
            fout.write('    {0:d}. `{1:s}`_\n'.format(index_count, files[file_type]))
            index_count += 1
    fout.write('\n')

    for file_type in ['hpp', 'cpp']:
        if files[file_type] is not None:
            fout.write("""
{0:s}
------

.. doxygenfile:: {1:s}

""".format(files[file_type], files[file_type])
            )
    fout.close()


def create_rst_files_for_cpp():
    """Generate all rst files."""
    cpp_files = get_cpp_files()
    create_cpp_tree(cpp_files)
    for file_base, files in cpp_files.iteritems():
        create_rst_file(file_base, files)


if __name__ == '__main__':
    create_rst_files_for_cpp()
