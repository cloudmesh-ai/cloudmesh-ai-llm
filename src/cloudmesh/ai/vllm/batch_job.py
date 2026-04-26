from cloudmesh.ai.vllm.config import VLLMConfig

class VLLMBatchJob:
    """
    Manages the creation of the batch job (sbatch or ijob) to execute the vLLM start script.
    """

    def __init__(self, config: VLLMConfig, script_path: str):
        self.config = config
        self.script_path = script_path
        self.name = config.name

    def generate_sbatch_content(self, working_dir: str) -> str:
        """
        Generate the content for a Slurm sbatch script.
        """
        partition = self.config.get('partition')
        reservation = self.config.get('reservation')
        
        content = f"""#!/bin/bash
#SBATCH --job-name=vllm-{self.name}
#SBATCH --output={working_dir}/{self.name}_%j.out
#SBATCH --error={working_dir}/{self.name}_%j.err
"""
        if partition:
            content += f"#SBATCH --partition={partition}\n"
        if reservation:
            content += f"#SBATCH --reservation={reservation}\n"
            
        content += f"""#SBATCH --gres=gpu:{self.config.get('gres', 'gpu:1')}
#SBATCH --cpus-per-task={self.config.get('cpus', '1')}
#SBATCH --mem={self.config.get('mem', '16G')}
#SBATCH --time={self.config.get('time', '24:00:00')}

{self.script_path}
"""
        return content

    def generate_ijob_command(self) -> str:
        """
        Generate the ijob command for interactive execution.
        """
        partition = self.config.get('partition')
        reservation = self.config.get('reservation')
        
        if not partition:
            raise ValueError(f"Partition is required for interactive (ijob) startup of server '{self.name}'. Please add 'partition' to the YAML config.")
        
        ijob_args = f"-p {partition}"
        if reservation:
            ijob_args += f" -r {reservation}"
            
        return f"ijob {ijob_args} {self.script_path}"

    def get_execution_command(self, mode: str, sbatch_script_path: str = None) -> str:
        """
        Return the command to execute the job based on the mode.
        
        Args:
            mode: 'sbatch' or 'ijob'
            sbatch_script_path: Path to the generated sbatch script (required for sbatch mode)
        """
        if mode == "sbatch":
            if not sbatch_script_path:
                raise ValueError("sbatch_script_path is required for sbatch mode")
            return f"sbatch {sbatch_script_path}"
        elif mode == "ijob":
            return self.generate_ijob_command()
        else:
            raise ValueError(f"Unsupported execution mode: {mode}")