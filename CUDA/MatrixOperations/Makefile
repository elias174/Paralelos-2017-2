TRANSFER=scp
REXEC=ssh
SSHFLAGS=-C # Compress data
WORKDIR_SERVER=CUDA/MatrixOperations
REMOTE=evalencia@manati:~/$(WORKDIR_SERVER)
CC=/usr/local/cuda/bin/nvcc
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
	$(REXEC) $(SSHFLAGS) evalencia@manati "cd $(WORKDIR_SERVER) && $(CC) $(SRC) && echo -e '$(script_content)' > script.sh \
		&& chmod +x script.sh"
	touch all-done

clean:
	rm -f *.ct all-done
