CFLAGSSIEVE = -O2 -g
CFLAGSMULT = -O2 -g 
ARCHTHUMB = -march=armv6
ARCHARMTHUMB = -march=armv7-a
CC = gcc

all:	sieve-arm sieve-thumb1 sieve-thumb2 mult-arm mult-thumb1 mult-thumb2

# Build and keep the assembly files so we can see them
sieve-thumb1.s : sieve.c makefile
	$(CC) $(CFLAGSSIEVE) -S -c -msoft-float $(ARCHTHUMB) -mthumb -o sieve-thumb1-temp.s sieve.c
	cat thumb1-hack sieve-thumb1-temp.s > sieve-thumb1.s
	rm sieve-thumb1-temp.s
sieve-thumb2.s : sieve.c makefile
	$(CC) $(CFLAGSSIEVE) -S -c $(ARCHARMTHUMB) -mthumb -o sieve-thumb2.s sieve.c
sieve-arm.s : sieve.c makefile
	$(CC) $(CFLAGSSIEVE) -S -c $(ARCHARMTHUMB) -o sieve-arm.s sieve.c
mult-thumb1.s : mult.c makefile
	$(CC) $(CFLAGSSIEVE) -S -c -msoft-float $(ARCHTHUMB) -mthumb -o mult-thumb1-temp.s mult.c
	cat thumb1-hack mult-thumb1-temp.s > mult-thumb1.s
	rm mult-thumb1-temp.s
mult-thumb2.s : mult.c makefile
	$(CC) $(CFLAGSSIEVE) -S -c $(ARCHARMTHUMB) -mthumb -o mult-thumb2.s mult.c
mult-arm.s : mult.c makefile
	$(CC) $(CFLAGSSIEVE) -S -c $(ARCHARMTHUMB) -o mult-arm.s mult.c

#Build the sieve executables
sieve-arm: sieve-arm.o sieve-main.c makefile
	$(CC) $(CFLAGSSIEVE) -marm $(ARCHARMTHUMB) -o sieve-arm sieve-arm.o sieve-main.c
sieve-thumb2: sieve-thumb2.o sieve-main.c makefile
	$(CC) $(CFLAGSSIEVE) -mthumb-interwork $(ARCHARMTHUMB) -o sieve-thumb2 sieve-thumb2.o sieve-main.c
sieve-thumb1: sieve-thumb1.o sieve-main.c makefile
	$(CC) $(CFLAGSSIEVE) -mthumb-interwork $(ARCHTHUMB) -o sieve-thumb1 sieve-thumb1.o sieve-main.c

#Build the sieve .o files
sieve-arm.o: sieve-arm.s makefile
	$(CC) $(CFLAGSSIEVE) -c $(ARCHARMTHUMB) -marm -o sieve-arm.o sieve.c
sieve-thumb2.o: sieve-thumb2.s makefile
	$(CC) $(CFLAGSSIEVE) -c $(ARCHARMTHUMB) -mthumb -o sieve-thumb2.o sieve.c
sieve-thumb1.o: sieve-thumb1.s makefile
	$(CC) $(CFLAGSSIEVE) -c $(ARCHTHUMB) -mthumb -o sieve-thumb1.o sieve-thumb1.s

#Build the mult executables
mult-arm: mult-arm.o mult-main.c makefile
	$(CC) $(CFLAGSMULT) -marm -o mult-arm mult-arm.o mult-main.c
mult-thumb2: mult-thumb2.o mult-main.c makefile
	$(CC) $(CFLAGSMULT) -mthumb-interwork -o mult-thumb2 mult-thumb2.o mult-main.c
mult-thumb1: mult-thumb1.o mult-main.c makefile
	$(CC) $(CFLAGSMULT) -mthumb-interwork -o mult-thumb1 mult-thumb1.o mult-main.c

#Build the mult .o files
mult-arm.o: mult-arm.s makefile
	$(CC) $(CFLAGSMULT) -c -marm -o mult-arm.o mult-arm.s
mult-thumb2.o: mult-thumb2.s makefile
	$(CC) $(CFLAGSMULT) -c -mthumb -o mult-thumb2.o mult-thumb2.s
mult-thumb1.o: mult-thumb1.s makefile
	$(CC) $(CFLAGSMULT) -c -mthumb -o mult-thumb1.o mult-thumb1.s
clean:
	rm sieve-arm sieve-thumb1 sieve-thumb2 mult-arm mult-thumb1 mult-thumb2 *.o *.s
