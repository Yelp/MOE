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

docs-no-tox:
		python docs/cpp_rst_maker.py
		doxygen docs/doxygen_config
		sphinx-apidoc -f -o docs moe
		sphinx-build -b html docs docs/_build/html

docs:
		tox -e docs
