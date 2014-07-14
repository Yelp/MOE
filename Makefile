all: production

clean:
		find . -name '*.pyc' -delete
		rm -rf moe/build

production:
		python setup.py install

test:
		testify -v moe.tests

style-test:
		pip install flake8 flake8-import-order pep8-naming flake8-docstrings pyflakes
		flake8 --ignore=E501,E126,E123,I101,I100,N806 moe
		pep257 moe
		pyflakes moe

docs:
		python docs/cpp_rst_maker.py
		doxygen docs/doxygen_config
		sphinx-apidoc -f -o docs moe
		sphinx-build -b html docs docs/_build/html
