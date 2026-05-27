import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.dml.color import RGBColor

def add_box(slide, left, top, width, height, text, bg_color, font_size=10, bold=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    fill = shape.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(*bg_color)
    line = shape.line
    line.color.rgb = RGBColor(100, 100, 100)
    line.width = Pt(1)
    tf = shape.text_frame
    tf.text = text
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = RGBColor(0, 0, 0)
    return shape

def add_arrow(slide, start_shape, end_shape, dashed=False):
    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, 0, 0, 0, 0)
    connector.line.color.rgb = RGBColor(50, 50, 50)
    connector.line.width = Pt(1.5)
    connector.line.end_arrowhead = 4 # TRIANGLE
    if dashed:
        connector.line.dash_style = 2 
    connector.begin_connect(start_shape, 2) # 2 = Bottom center
    connector.end_connect(end_shape, 0)     # 0 = Top center
    return connector

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])

# --- 1. LOCAL WORKSTATION ---
workstation = add_box(slide, 2.5, 0.2, 8.5, 3.2, "Local Workstation", (245, 245, 245), font_size=12, bold=True)
env = add_box(slide, 5.75, 0.4, 2, 0.4, ".config/cloudmesh/.env", (255, 250, 200))
cli = add_box(slide, 3.5, 1.0, 6.5, 0.8, "llmctl Unified CLI", (255, 255, 255))
l_launch = add_box(slide, 3.7, 1.3, 1.2, 0.4, "llmctl launch", (220, 240, 255))
l_proxy = add_box(slide, 5.1, 1.3, 1.2, 0.4, "llmctl proxy", (220, 240, 255))
l_check = add_box(slide, 6.5, 1.3, 1.2, 0.4, "llmctl check", (220, 240, 255))
l_tunnel = add_box(slide, 7.9, 1.3, 1.2, 0.4, "llmctl tunnel", (220, 240, 255))
lite_llm = add_box(slide, 5.1, 2.4, 1.5, 0.6, "LiteLLM Proxy\nPort 4000", (230, 210, 240))
uva_ssh = add_box(slide, 7.9, 2.4, 1.5, 0.6, "UVA SSH Tunnel\nPort 8080", (230, 210, 240))

# --- 2. REMOTE INFRASTRUCTURE ---
remote = add_box(slide, 2.5, 4.2, 8.5, 2.8, "Remote Infrastructure", (250, 250, 230), font_size=12, bold=True)
hosts = add_box(slide, 3.0, 4.8, 3.0, 1.8, "Remote GPU Hosts", (255, 255, 255))
h_white = add_box(slide, 3.2, 5.2, 1.2, 0.4, "Host: white", (255, 250, 200))
h_spark = add_box(slide, 4.6, 5.2, 1.2, 0.4, "Host: spark", (255, 250, 200))
vllm = add_box(slide, 3.2, 5.8, 2.6, 0.6, "vLLM Containers\nPorts 18000+", (230, 210, 240))
kimi = add_box(slide, 7.5, 5.0, 2.5, 0.6, "UVA GENAI Kimi K2.5\nRemote API", (230, 210, 240))

# --- 3. CONNECTORS ---
add_arrow(slide, env, cli)
add_arrow(slide, l_proxy, lite_llm)
add_arrow(slide, l_check, lite_llm)
add_arrow(slide, l_tunnel, uva_ssh)
add_arrow(slide, l_launch, hosts, dashed=True)
add_arrow(slide, lite_llm, vllm)
add_arrow(slide, lite_llm, kimi)
add_arrow(slide, uva_ssh, kimi, dashed=True)

prs.save("Architecture_Diagram_V3.pptx")
print("Success! Created Architecture_Diagram_V3.pptx")