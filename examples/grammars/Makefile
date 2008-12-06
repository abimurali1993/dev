# NLTK: Documentation Makefile
#
# Copyright (C) 2001-2008 NLTK Project
# Author: Ewan Klein <ewan@inf.ed.ac.uk>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT

PUBLISH = ../../../nltk_data/packages/grammars

PACKAGE_DIRS = book_grammars sample_grammars basque_grammars spanish_grammars
PACKAGES := $(addsuffix .zip, $(PACKAGE_DIRS))

NLTK_VERSION = $(shell python -c 'import nltk; print nltk.__version__')
NLTK_URL = $(shell python -c 'import nltk; print nltk.__url__')

ZIP = zip

all: clean publish

zip: $(PACKAGES)

clean:	
	rm *.zip

%.zip: %
	$(ZIP) -r $< $<  -x \*.svn*

publish: zip
	cp $(PACKAGES) $(PUBLISH)