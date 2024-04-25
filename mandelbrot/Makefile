all:
	nvcc main.cu -lraylib -lGL -lm -lpthread -ldl -lrt -lX11 -allow-unsupported-compiler -o main
	./main
clean:
	rm main
