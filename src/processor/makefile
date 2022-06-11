DIR_INC += -I../include

OBJS = bin/main.o
CFLAGS = -g $(DIR_INC) -std=c++17 -Wno-narrowing -Werror=vla -MD -I/usr/include/opencv4
LIBS = -lopencv_core -lopencv_dnn

default: bin/processor

bin/%.o: %.cpp
	[ -d bin ] || mkdir -p bin
	g++ $(CFLAGS) -c $< -o $@

bin/processor: $(OBJS)
	g++ -g $^ $(LIBS) -o $@

clean:
	-rm -rf bin/*.o
	-rm -rf bin/*.d
	-rm -rf bin/processor

test: bin/processor
#	grep -n --color TODO *.*
	echo
	cd bin && exec gdb -q -ex 'set print thread-events off' -ex run ./processor

-include $(OBJS:.o=.d)
