#FLAGS= -DDEBUG
#LIBS= -lm
#ALWAYS_REBUILD=makefile

#nbody: nbody.o compute.o
#	nvcc $(FLAGS) $^ -o $@ $(LIBS)
#nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
#	gcc $(FLAGS) -c -g $< 
#compute.o: cuda_compute.cu config.h vector.h $(ALWAYS_REBUILD)
#	nvcc $(FLAGS) -c -g $<
#clean:
#	rm -f *.o nbody

FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $<
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $<
clean:
	rm -f *.o nbody
