  # Install Docker using convenience script
  curl -fsSL https://get.docker.com -o get-docker.sh
  sh get-docker.sh
  
  # Start Docker daemon
  dockerd > /var/log/dockerd.log 2>&1 &
  
  # Wait for Docker
  until docker info > /dev/null 2>&1; do
    echo "Waiting for Docker daemon..."
    sleep 2
  done
  
  echo "Docker is ready!"
  docker --version