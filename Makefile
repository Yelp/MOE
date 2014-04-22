SUBDIRS=moe/build
TESTIFY=testify

all: production

production:
		for SUBDIR in $(SUBDIRS); do if [ -e $$SUBDIR/Makefile ]; then ($(MAKE) -C $$SUBDIR $(MFLAGS)); fi; done

test:
		$(TESTIFY) -v moe.tests

docs:
		$(MAKE) -C doc doxygen
		sphinx-apidoc -f -o doc moe
		$(MAKE) -C doc html
