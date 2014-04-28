SUBDIRS=moe/build
TESTIFY=testify

all: production

clean:
		find . -name '*.pyc' -delete

production:
		for SUBDIR in $(SUBDIRS); do if [ -e $$SUBDIR/Makefile ]; then ($(MAKE) -C $$SUBDIR $(MFLAGS)); fi; done

test:
		$(TESTIFY) -v moe.tests

docs:
		$(MAKE) -C docs doxygen
		sphinx-apidoc -f -o docs moe
		$(MAKE) -C docs html
