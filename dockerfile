# WSBIM2243 DOCKERFILE 
# @AUTHOR AMAN 

# Get image
FROM ubuntu:latest

# Install python3
RUN apt-get update && apt-get install -y python3 \
    python3-pip
    
# Install git to get latest version of the project preinstalled
RUN apt-get install -y git

# Install packages for the project
RUN pip3 install -r requirements.txt

# Get work in github
RUN git clone https://github.com/amanpreetsingh-BE/DeepRL-Protontherapy