make:
	nvcc main.cpp kernel.cu funcs.cpp -o flameGPU.exe

clean:
	rm -f -r *.exe *~ *.vtk
