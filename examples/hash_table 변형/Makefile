##########################
###### flag setting ######
debug = 1
##########################

CC = g++
CFLAGS = -O2
TARGET = main.out
OBJS = main.o hash.o

ifeq ($(debug), 1)
    CFLAGS += -g
endif

$(TARGET) : $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

clean :
	rm -f *.o main.out
