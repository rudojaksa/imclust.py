T := imclust.py

all: $T

clean:
	rm -f index.html
	rm -f README.md
	rm -fr __pycache__

README.md: $T
	$< -h | man2md > $@

-include ~/.github/Makefile.git
