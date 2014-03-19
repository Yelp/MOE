SUBDIRS=build
PYTHON=PYTHONPATH="$(shell pwd)" python
TESTIFY=YELPCODE="$(shell pwd)" testify

all: production

production:
		for SUBDIR in $(SUBDIRS); do if [ -e $$SUBDIR/Makefile ]; then ($(MAKE) -C $$SUBDIR $(MFLAGS)); fi; done

test:
		$(TESTIFY) -v tests

docs:
		$(MAKE) -C doc doxygen
		$(MAKE) -C doc html
