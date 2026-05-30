import time
from rich.table import Table
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Header, Footer, Static, Label
from textual.containers import Container, Grid
from cloudmesh.ai.common.io import console

class RenderVLLMTable:
    """Helper class to render vLLM configurations into a Textual DataTable."""
    
    @staticmethod
    def render(table: DataTable, servers: dict):
        table.cursor_type = "row"
        table.add_columns(
            "Service Name", "Host", "Model", "Image", 
            "TP Size", "GPU Util", "Port", "Account", 
            "Partition", "Reservation", "GRES", "CPUs", "Mem"
        )

        if isinstance(servers, dict):
            for name, config in servers.items():
                if isinstance(config, dict):
                    table.add_row(
                        name,
                        config.get("host", "N/A"),
                        config.get("model", "N/A"),
                        config.get("image", "N/A"),
                        str(config.get("tensor_parallel_size", "N/A")),
                        str(config.get("gpu_memory_utilization", "N/A")),
                        str(config.get("port", "8000")),
                        config.get("account", "N/A"),
                        config.get("partition", "N/A"),
                        config.get("reservation", "N/A"),
                        config.get("gres", "N/A"),
                        str(config.get("cpus", "N/A")),
                        config.get("mem", "N/A"),
                    )

class VLLMServiceSelector(App):
    """Textual App for selecting a vLLM service."""
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, servers):
        super().__init__()
        self.servers = servers
        self.selected_service = None
        self.selected_host = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable()
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        RenderVLLMTable.render(table, self.servers)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = self.query_one(DataTable)
        row_data = table.get_row(event.row_key)
        self.selected_service = row_data[0]
        self.selected_host = row_data[1]
        self.exit()

class LLMMonitorApp(App):
    """Textual App for real-time vLLM monitoring (Grafana-like)."""
    CSS = """
    Grid {
        grid-size: 2 2;
        grid-gutter: 1;
        padding: 1;
    }
    .panel {
        border: solid green;
        padding: 1;
        height: 100%;
        width: 100%;
    }
    .panel-title {
        text-style: bold;
        color: cyan;
        margin-bottom: 1;
    }
    .metric-value {
        text-style: bold;
        color: magenta;
    }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, client, server_name):
        super().__init__()
        self.client = client
        self.server_name = server_name
        self.tps_history = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Grid():
            with Static(classes="panel", id="throughput_panel"):
                yield Label("Throughput", classes="panel-title")
                yield Static("0.00 tokens/sec", id="tps_val", classes="metric-value")
                yield Static("", id="tps_chart")
            with Static(classes="panel", id="tokens_panel"):
                yield Label("Token Counts", classes="panel-title")
                yield Static("Prompt: 0", id="prompt_val")
                yield Static("Generation: 0", id="gen_val")
                yield Static("Total: 0", id="total_val")
            with Static(classes="panel", id="requests_panel"):
                yield Label("Request Queue", classes="panel-title")
                yield Static("Running: 0", id="run_val")
                yield Static("Swapped: 0", id="swap_val")
            with Static(classes="panel", id="cache_panel"):
                yield Label("GPU KV Cache", classes="panel-title")
                yield Static("0.00%", id="cache_val", classes="metric-value")
        yield Footer()

    def on_mount(self) -> None:
        self.prev_tokens = 0
        self.prev_time = time.time()
        self.set_interval(1, self.update_metrics)

    def update_metrics(self) -> None:
        metrics = self.client.get_metrics()
        if metrics:
            prompt = metrics.get("vllm:prompt_tokens_total", 0)
            gen = metrics.get("vllm:generation_tokens_total", 0)
            run = metrics.get("vllm:num_requests_running", 0)
            swap = metrics.get("vllm:num_requests_swapped", 0)
            cache = metrics.get("vllm:gpu_cache_usage_perc", 0) * 100
            
            now = time.time()
            dt = now - self.prev_time
            total = prompt + gen
            tps = (total - self.prev_tokens) / dt if self.prev_tokens > 0 else 0
            self.prev_tokens = total
            self.prev_time = now

            self.tps_history.append(tps)
            if len(self.tps_history) > 40:
                self.tps_history.pop(0)

            # Update UI
            self.refresh_ui(tps, prompt, gen, total, run, swap, cache)

    def refresh_ui(self, tps, prompt, gen, total, run, swap, cache):
        self.query_one("#tps_val").update(f"{tps:.2f} tokens/sec")
        self.query_one("#prompt_val").update(f"Prompt: {int(prompt):,}")
        self.query_one("#gen_val").update(f"Generation: {int(gen):,}")
        self.query_one("#total_val").update(f"Total: {int(total):,}")
        self.query_one("#run_val").update(f"Running: {int(run)}")
        self.query_one("#swap_val").update(f"Swapped: {int(swap)}")
        self.query_one("#cache_val").update(f"{cache:.2f}%")
        
        # Update sparkline
        self.query_one("#tps_chart").update(self.generate_sparkline())

    def generate_sparkline(self):
        if not self.tps_history: return ""
        chars = " ▂▃▄▅▆▇█"
        mini, maxi = min(self.tps_history), max(self.tps_history)
        diff = maxi - mini
        if diff == 0: return chars[0] * len(self.tps_history)
        return "".join([chars[int(((v - mini) / diff) * 7)] for v in self.tps_history])

def select_vllm_service(db, group_filter=None):
    """Interactively select a vLLM service from the available configurations using Textual."""
    # Handle both DotDict and objects with .get()
    if hasattr(db, 'get'):
        servers = db.get("cloudmesh.ai.server", {})
    else:
        servers = {}
    
    if not servers:
        console.error("No vLLM server configurations found in the config file.")
        return None, None, None

    app = VLLMServiceSelector(servers)
    app.run()
    
    return app.selected_service, None, app.selected_host