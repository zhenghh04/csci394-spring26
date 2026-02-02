# HPL build helper (template). Adjust paths and libraries for your cluster.

all: build

# Wrapper so students can just run: make build
build:
	./build_hpl.sh

clean:
	@if [ -d hpl-* ]; then \\
		for d in hpl-*; do \\
			$(MAKE) -C $$d arch=UNKNOWN clean || true; \\
		done; \\
	fi

distclean:
	@rm -rf hpl-* hpl-*.tar.gz Make.* || true

.PHONY: all build clean distclean
