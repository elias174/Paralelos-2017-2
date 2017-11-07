TRANSFER=scp
REXEC=ssh
SSHFLAGS=-C # Compress data
WORKDIR_SERVER=CUDA/ScaleGrise
REMOTE=evalencia@manati:~/$(WORKDIR_SERVER)
#REMOTE=eliasubuntu@192.168.0.12:~/$(WORKDIR_SERVER)
CC=/usr/local/cuda/bin/nvcc
FLAGS_CC = -I /opt/shared/opencv-2.4.13/include -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -L/opt/shared/opencv-2.4.13/lib
FILES=main.ct
SRC=main.cu
EXEC_NAME=vector_addition
NODES=1
PPN=1
GPUS=1
dollar := $$

define script_content = 
#!/bin/bash\n#PBS -N test_$(EXEC_NAME)\n#PBS -l nodes=$(NODES):ppn=$(PPN):gpus=$(GPUS)\ncd ~/$(WORKDIR_SERVER)\n./a.out
endef


%.ct : %.cu
	$(TRANSFER) $(SSHFLAGS) $< $(REMOTE)
	touch $@


all-done: $(FILES)
	$(REXEC) $(SSHFLAGS) evalencia@manati "cd $(WORKDIR_SERVER) && $(CC) $(SRC) $(FLAGS_CC) && echo -e '$(script_content)' > script.sh \
		&& chmod +x script.sh"
	touch all-done

clean:
	rm -f *.ct all-done
