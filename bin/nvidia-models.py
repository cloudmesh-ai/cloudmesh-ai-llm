import json
import pandas as pd
import os

def generate_model_html_report(json_file, output_html):
    # 1. Validation
    if not os.path.exists(json_file) or os.path.getsize(json_file) == 0:
        print(f"Error: {json_file} is missing. Run your NGC CLI command first.")
        return

    # 2. Load and Process Data
    with open(json_file, 'r') as f:
        data = json.load(f)

    processed_data = []
    for entry in data:
        processed_data.append({
            "Name": entry.get('displayName', 'N/A'),
            "Version": entry.get('latestVersionIdStr', 'N/A'),
            "Size (GB)": round(entry.get('latestVersionSizeInBytes', 0) / 1e9, 2),
            "Updated": entry.get('updatedDate', 'N/A')[:10],
            "Description": entry.get('description', 'N/A')
        })

    df = pd.DataFrame(processed_data)

    # 3. HTML Generation (with Paging disabled explicitly)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NVIDIA Model Catalog</title>
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
        <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <style>
            body {{ font-family: sans-serif; padding: 20px; background-color: #f4f7f6; }}
            .container {{ background-color: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            h2 {{ color: #76b900; }}
            /* Force pagination elements to be hidden */
            .dataTables_paginate {{ display: none !important; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>NVIDIA Model Catalog (Full List)</h2>
            {df.to_html(classes='display', index=False, border=0)}
        </div>
        <script>
            $(document).ready(function() {{
                $('table').DataTable({{
                    "paging": false,
                    "scrollY": "70vh",
                    "scrollCollapse": true,
                    "searching": true
                }});
            }});
        </script>
    </body>
    </html>
    """

    # 4. Save
    with open(output_html, 'w') as f:
        f.write(html_content)
    print(f"Success! Open {output_html} in your browser.")

if __name__ == "__main__":
    generate_model_html_report('nvidia-model-catalog.json', 'model_report.html')