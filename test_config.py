import yaml
from cloudmesh.ai.vllm.config import VLLMConfig
from cloudmesh.ai.vllm.script import VLLMScript
from cloudmesh.ai.common.io import console

def test_vllm_config():
    """Test the VLLMConfig and VLLMScript classes for uva.gemma."""
    
    name = "uva.gemma"
    console.banner(f"Testing Configuration for: {name}")
    
    try:
        config = VLLMConfig()
        
        # 1. Dump the full merged global configuration using VLLMConfig.yaml property
        console.banner("Full Merged Global Configuration")
        print(config.yaml)
        
        if not config:
            console.warning(f"No configuration found for {name}")
            return

        # 2. Dump the expanded server-specific configuration using VLLMConfig.yaml_data property
        console.banner(f"Expanded Server Configuration for {name}")
        print(config.yaml_data)
        
        # Test DotDict access (direct attribute access)
        if "cloudmesh" in config:
            console.ok(f"DotDict access test: Root 'cloudmesh' key found in global config.")
        
        # Test Script Generation
        console.banner(f"Generating Launch Script for {name}")
        
        # Use the full config path as requested
        config_path = f"cloudmesh.ai.server.{name}"
        script_gen = VLLMScript(config, config_path)
        script_content = script_gen.generate()
        console.print(script_content)
        console.print("\n" + "="*60 + "\n")

        # Test Server Retrieval
        console.banner(f"Testing Server Retrieval for {name}")
        server_config = config.get_server(name)
        console.print(f"Server Config Type: {type(server_config)}")
        console.print(f"Server Config Content: {server_config}")
        
        if server_config:
            host = server_config.get("host")
            if host:
                console.ok(f"Found host: {host}")
            else:
                console.error("Host key NOT found in server config!")
        else:
            console.error(f"Could not retrieve server config for {name}")
            
        console.print("\n" + "="*60 + "\n")

        # Test Expansion of Scoped Config
        console.banner(f"Testing Expansion for {name}")
        if server_config:
            expanded_config = config.expand_external_references(server_config)
            console.print("Expanded Server Config:")
            console.print(expanded_config)
            
            # Verify a specific expanded value (e.g., user or host if it was a placeholder)
            expanded_host = expanded_config.get("host")
            console.ok(f"Expanded host: {expanded_host}")
        else:
            console.error("Cannot test expansion without server config")
            
        console.print("\n" + "="*60 + "\n")
        print (config.keys())

    except Exception as e:
        console.error(f"Error testing {name}: {e}")

if __name__ == "__main__":
    test_vllm_config()