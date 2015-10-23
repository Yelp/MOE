all: production

clean:
		find . -name '*.pyc' -delete
		rm -rf moe/build
		rm -rf build
		rm -rf MOE.egg-info

production:
		python setup.py install

test:
		py.test -v moe/tests moe_examples/tests

style-test:
		pip install flake8 flake8-import-order pep8-naming flake8-docstrings pyflakes
		flake8 --ignore=E501,E126,E123,I101,I100,N806 moe
		pep257 moe
		pyflakes moe
		flake8 --ignore=E501,E126,E123,I101,I100,N806 moe_examples
		pep257 moe_examples
		pyflakes moe_examples

docs:
		python docs/cpp_rst_maker.py
		doxygen docs/doxygen_config
		sphinx-apidoc -f -o docs moe
		sphinx-apidoc -f -T -o docs moe_examples
		sphinx-build -b html docs docs/_build/html
