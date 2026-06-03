.PHONY: image

image:
	module load apptainer; apptainer build mock.sif mock.def

run: 
	apptainer run mock.sif --port 20001

cp:
	rsync -r ../mock uva:/scratch/thf2bn/.