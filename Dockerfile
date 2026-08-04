FROM anibali/pytorch:2.0.1-cuda11.8-ubuntu22.04

# Optional: set working directory
WORKDIR /app

# Set up time zone
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Copy requirements first for caching
COPY ./requirements.txt .

# Install system and Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy only necessary files for runtime
COPY ./infer_convexadam.py /app/
COPY ./MIR /app/MIR
RUN sudo chmod -R a+rw /app/MIR
# Install the MIR package (editable if you plan to modify)
WORKDIR /app/MIR
RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/input /app/output
RUN chmod -R 777 /app/output
# Switch back to /app as the working directory
WORKDIR /app

# ANTs affine + ConvexAdam-MIND (SVF) registration - one moving/fixed PET+CT set per run.
# The registration is optimization-based (no learned weights) and the container makes no
# assumption about filenames: the caller passes five paths - fixed CT, fixed PET, moving CT,
# moving PET, output. It writes one original-resolution displacement field, composed from
# the ANTs affine and the ConvexAdam deformable fields. (CT drives the field; PET is a
# required input mirroring the participant PET+CT interface and is used for the QA preview.)
#
#   docker run ... <image>  /app/input/FIXED_CT.nii.gz  /app/input/FIXED_PET.nii.gz \
#                           /app/input/MOVING_CT.nii.gz /app/input/MOVING_PET.nii.gz \
#                           /app/output/DISP.nii.gz
ENTRYPOINT ["python3","-u","/app/infer_convexadam.py"]
