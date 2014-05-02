"""Automatically create *.rst files for sphinx for all doxygen+breathe created C++ api docs."""
from os import listdir
from os.path import isfile, join

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
            file_base = file_name.split('.')[0]
            file_ext = file_name.split('.')[1]
            if file_base not in cpp_files and file_ext in ['hpp', 'cpp']:
                cpp_files[file_base] = {
                        'cpp': None,
                        'hpp': None,
                        }
            if file_ext == 'hpp':
                cpp_files[file_base]['hpp'] = file_name
            elif file_ext == 'cpp':
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
        fout.write('    %s.rst\n' % cpp_file)
    fout.close()

def create_rst_file(file_base, files):
    """Create a default file_base.rst file for sphinx."""
    fout = open(
            join(
                'docs',
                '%s.rst' % file_base,
                ),
            'w'
            )
    fout.write(
"""%s
=====

**Contents:**

""" % file_base
    )
    index_count = 1
    for file_type in ['hpp', 'cpp']:
        if files[file_type] is not None:
            fout.write('    %d. `%s`_\n' % (index_count, files[file_type]))
            index_count += 1
    fout.write('\n')

    for file_type in ['hpp', 'cpp']:
        if files[file_type] is not None:
            fout.write("""
%s
------

.. doxygenfile:: %s

""" % (files[file_type], files[file_type])
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
