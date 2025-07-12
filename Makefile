# Tag for the Docker images
DOCKER_TAG = latest

# Components and corresponding Dockerfiles
COMPONENTS = load preprocess train evaluate deploy predict 

# Directory where Dockerfiles are located
DOCKERFILES_DIR = dockerfiles

# Target to build all images
all: $(COMPONENTS)

# Build Docker images for each component
$(COMPONENTS):
	@echo "Building Docker image for $@..."
	@docker build -t $@:$(DOCKER_TAG) -f $(DOCKERFILES_DIR)/Dockerfile_$@ .	
	@echo "Docker image for $@ built successfully!"

# Clean up images (optional)
clean:
	@echo "Removing all built images..."
	@docker rmi yourusername/$(COMPONENTS):$(DOCKER_TAG)

# List all Docker images (optional)
list:
	@docker images

