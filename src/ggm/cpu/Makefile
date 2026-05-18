CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fPIC -Wall -Wextra -std=c11
LDFLAGS ?= -shared

SOURCES = aes_ref.c aes_ni.c aes_omp.c spongent_ref.c spongent_omp.c
OBJECTS = $(SOURCES:.c=.o)

libggmcpu.so: $(OBJECTS)
	$(CC) $(LDFLAGS) -fopenmp $^ -o $@

aes_ni.o: aes_ni.c ggmcpu.h
	$(CC) $(CFLAGS) -maes -msse4.1 -c $< -o $@

aes_omp.o: aes_omp.c ggmcpu.h
	$(CC) $(CFLAGS) -fopenmp -c $< -o $@

spongent_omp.o: spongent_omp.c ggmcpu.h
	$(CC) $(CFLAGS) -fopenmp -c $< -o $@

%.o: %.c ggmcpu.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o libggmcpu.so

.PHONY: clean
