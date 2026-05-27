import subprocess
from cloudmesh.ai.common.io import console
from cloudmesh.ai.common.sys import os_is_mac
import time
import os

class DockerManager:
    """Handles Docker operations and lifecycle management."""

    def check_docker(self):
        """Verify that Docker is running. On macOS, offer to start it."""
        try:
            subprocess.run(["docker", "info"], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            if os_is_mac():
                if console.ynchoice("Docker is not running. Would you like to start Docker Desktop?", default=True):
                    console.print("Starting Docker Desktop...")
                    subprocess.run(["open", "-a", "Docker"], capture_output=True)
                    return self._wait_for_docker()
            
            console.error("Docker is not running or could not be found.")
            console.print("Please start the Docker Desktop application and try again.")
            return False

    def _wait_for_docker(self):
        """Poll docker info until it becomes available."""
        console.print("Waiting for Docker to start...", end="", flush=True)
        for _ in range(30):  # Wait up to 30 seconds
            try:
                subprocess.run(["docker", "info"], check=True, capture_output=True)
                console.print("\n")
                console.ok("Docker is now available!")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(".", end="", flush=True)
                time.sleep(1)
        
        console.print("\n")
        console.error("Docker failed to start within 30 seconds.")
        return False

    def stop_container(self, container_name: str):
        """Stop and remove a specific Docker container."""
        console.print(f"Stopping existing container {container_name} if it exists...")
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)

    def run_container(self, cmd: str):
        """Execute a docker run command."""
        try:
            subprocess.run(cmd, check=True, capture_output=True, shell=True)
            return True
        except subprocess.CalledProcessError as e:
            console.error(f"Error launching container: {e.stderr.decode()}")
            return False

    def run_with_env_file(
        self,
        image: str,
        container_name: str,
        env_file: str = None,
        ports: dict = None,
        volumes: dict = None,
        command: str = None,
        detach: bool = True,
        remove: bool = True,
        **kwargs
    ):
        """
        Run a Docker container with environment variables from a file.
        
        Args:
            image (str): Docker image name
            container_name (str): Name for the container
            env_file (str, optional): Path to .env file
            ports (dict, optional): Port mappings {host_port: container_port}
            volumes (dict, optional): Volume mappings {host_path: container_path}
            command (str, optional): Command to run in container
            detach (bool): Run in detached mode
            remove (bool): Remove container after stop
            **kwargs: Additional docker run arguments
        
        Returns:
            bool: True if container started successfully
        """
        # Build the docker run command
        cmd_parts = ["docker", "run"]
        
        if detach:
            cmd_parts.append("-d")
        if remove:
            cmd_parts.append("--rm")
        
        cmd_parts.extend(["--name", container_name])
        
        # Add env file
        if env_file and os.path.exists(env_file):
            cmd_parts.extend(["--env-file", env_file])
        
        # Add port mappings
        if ports:
            for host_port, container_port in ports.items():
                cmd_parts.extend(["-p", f"{host_port}:{container_port}"])
        
        # Add volume mappings
        if volumes:
            for host_path, container_path in volumes.items():
                cmd_parts.extend(["-v", f"{host_path}:{container_path}"])
        
        # Add additional kwargs as environment variables
        for key, value in kwargs.items():
            if value is not None:
                cmd_parts.extend(["-e", f"{key.upper()}={value}"])
        
        # Add image
        cmd_parts.append(image)
        
        # Add command
        if command:
            cmd_parts.append(command)
        
        cmd = " ".join(cmd_parts)
        console.print(f"Running: {cmd}")
        
        return self.run_container(cmd)

    def run_compose_with_env(
        self,
        compose_file: str,
        env_file: str = None,
        service: str = None,
        detach: bool = True,
        build: bool = False
    ):
        """
        Run docker-compose with environment file support.
        
        Args:
            compose_file (str): Path to docker-compose.yml
            env_file (str, optional): Path to .env file
            service (str, optional): Specific service to run
            detach (bool): Run in detached mode
            build (bool): Build before running
        
        Returns:
            bool: True if successful
        """
        cmd_parts = ["docker-compose", "-f", compose_file]
        
        # Add env file if provided
        if env_file and os.path.exists(env_file):
            # For docker-compose, env file is typically in the same directory
            # and loaded automatically if named .env, but we can also use --env-file
            cmd_parts.extend(["--env-file", env_file])
        
        if build:
            cmd_parts.append("--build")
        
        cmd_parts.append("up")
        
        if detach:
            cmd_parts.append("-d")
        
        if service:
            cmd_parts.append(service)
        
        cmd = " ".join(cmd_parts)
        console.print(f"Running: {cmd}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, shell=True)
            return True
        except subprocess.CalledProcessError as e:
            console.error(f"Error with docker-compose: {e.stderr.decode()}")
            return False

    def generate_docker_env_file(self, config: dict, output_path: str = ".env"):
        """
        Generate a .env file for Docker from a configuration dictionary.
        
        Args:
            config (dict): Configuration dictionary
            output_path (str): Where to write the .env file
        
        Returns:
            str: Path to the generated file
        """
        lines = ["# Docker Environment Configuration", "# Generated by DockerManager", ""]
        
        # Flatten config dict to env vars
        def flatten_dict(d, prefix=""):
            for key, value in d.items():
                env_key = f"{prefix}{key.upper()}" if prefix else key.upper()
                
                if isinstance(value, dict):
                    flatten_dict(value, f"{env_key}_")
                elif isinstance(value, (list, tuple)):
                    lines.append(f"{env_key}={','.join(str(v) for v in value)}")
                else:
                    lines.append(f"{env_key}={value}")
        
        flatten_dict(config)
        
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        
        console.print(f"[green]Generated Docker .env file: {output_path}[/green]")
        return output_path

    def stop_compose(self, compose_file: str):
        """
        Stop docker-compose services.
        
        Args:
            compose_file (str): Path to docker-compose.yml
        
        Returns:
            bool: True if successful
        """
        cmd = f"docker-compose -f {compose_file} down"
        try:
            subprocess.run(cmd, check=True, capture_output=True, shell=True)
            return True
        except subprocess.CalledProcessError as e:
            console.error(f"Error stopping docker-compose: {e.stderr.decode()}")
            return False
