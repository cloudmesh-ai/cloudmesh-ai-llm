import os
from cloudmesh.ai.common.ssh import SSHConfig
from .config import VLLMConfig

class IJob:
    """Helper class to manage ijob command construction and validation."""
    def __init__(self, db):
        self.db = db
        self.config = None
        self.name = None

    def get(self, group, name):
        """Reads the needed info from the yaml file."""
        self.name = name
        self.config = VLLMConfig()
        return self

    def username(self):
        """Returns the username for the remote host from ssh config, falling back to local user."""
        host = self.host()
        return SSHConfig().username(host) if host else os.environ.get("USER", "user")

    def name(self):
        """Returns the name for the job, replacing $USER with the actual username."""
        if not self.name:
            return None
        return self.name.replace("$USER", self.username())

    def host(self):
        """Returns the host name from the configuration."""
        return self.config.get("host")

    def check(self):
        """Verifies that all mandatory information for the ijob is present."""
        mandatory = ["allocation", "partition", "time", "gres"]
        missing = [field for field in mandatory if not self.config.get(field)]
        if missing:
            raise ValueError(f"Missing mandatory ijob parameters in config: {', '.join(missing)}")
        return True

    def command(self):
        """Returns the ijob command string."""
        allocation = self.config.get("allocation")
        partition = self.config.get("partition")
        time = self.config.get("time")
        gres = self.config.get("gres")
        reservation = self.config.get("reservation")

        args = [
            "-c 1",
            f"-A {allocation}",
            f"-p {partition}",
            f"--time={time}",
            f"--gres={gres}"
        ]
        if reservation:
            args.append(f"--reservation={reservation}")
        
        return f"ijob {' '.join(args)}"

    def yaml(self):
        """Returns all the data from the yaml file."""
        return self.config
