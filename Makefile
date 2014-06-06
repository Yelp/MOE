all: production

clean:
		find . -name '*.pyc' -delete
		rm -rf moe/build

production:
		python setup.py install

test-no-tox:
		testify -v moe.tests

test:
		tox
		tox -e pep8

style-test-no-tox:
		pip install flake8 flake8-import-order pep8-naming flake8-docstrings
		flake8 --ignore=E501,E126,E123,I101,N806 moe
		pep257 moe

style-test:
		tox -e pep8

docs-no-tox:
		python docs/cpp_rst_maker.py
		doxygen docs/doxygen_config
		sphinx-apidoc -f -o docs moe
		sphinx-build -b html docs docs/_build/html

docs:
		tox -e docs
