TRANSFER=scp
REXEC=ssh
SSHFLAGS=-C # Compress data
WORKDIR_SERVER=TrapecioReduce
REMOTE=evalencia@manati:~/$(WORKDIR_SERVER)
CC=/opt/shared/openmpi-2.0.1/bin/mpicc
MPI_RUN=/opt/shared/openmpi-2.0.1/bin/mpirun
FILES=main.ct
SRC=main.c
EXEC_NAME=trapecio_reduce
dollar := $$

define script_content = 
#!/bin/bash\n#PBS -N test_$(EXEC_NAME)\n#PBS -l nodes=$(NODES)\ncd /home/evalencia/$(WORKDIR_SERVER)\n$(MPI_RUN) -n $(NODES) ./a.out
endef


%.ct : %.c
	$(TRANSFER) $(SSHFLAGS) $< $(REMOTE)
	touch $@


all-done: $(FILES)
	$(REXEC) $(SSHFLAGS) evalencia@manati "cd $(WORKDIR_SERVER) && $(CC) $(SRC) && echo -e '$(script_content)' > script.sh \
		&& chmod +x script.sh"
	touch all-done

clean:
	rm -f *.ct all-done
