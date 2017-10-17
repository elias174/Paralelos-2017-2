TRANSFER=scp
REXEC=ssh
SSHFLAGS=-C # Compress data
WORKDIR_SERVER=MatrixVector
REMOTE=evalencia@manati:~/$(WORKDIR_SERVER)
CC=g++
FLAGS=-fopenmp
FILES=main.ct
SRC=main.cpp
EXEC_NAME=matrix_vector
NODES=1
dollar := $$
ROWS = 8000000
COLUMNS = 8
THREADS = 1

define script_content = 
#!/bin/bash\n#PBS -N test_$(EXEC_NAME)\n#PBS -l nodes=$(NODES)\n./a.out $(ROWS) $(COLUMNS) $(THREADS)
endef


%.ct : %.cpp
	$(TRANSFER) $(SSHFLAGS) $< $(REMOTE)
	touch $@


all-done: $(FILES)
	$(REXEC) $(SSHFLAGS) evalencia@manati "cd $(WORKDIR_SERVER) && $(CC) $(SRC) $(FLAGS) && echo -e '$(script_content)' > script.sh \
		&& chmod +x script.sh"
	touch all-done

clean:
	rm -f *.ct all-done
