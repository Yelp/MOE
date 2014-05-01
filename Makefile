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

docs:
		tox -e docs
