#!/usr/bin/env python3
"""Compile Markdown blog post to PDF with elegant styling."""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path

# Read the markdown file
md_path = Path(__file__).parent / "dova-agentic-fabric.md"
md_content = md_path.read_text()

# Remove YAML frontmatter
if md_content.startswith("---"):
    parts = md_content.split("---", 2)
    if len(parts) >= 3:
        md_content = parts[2]

# Convert markdown to HTML
md = markdown.Markdown(extensions=[
    'fenced_code',
    'tables',
    'toc',
    'codehilite',
])
html_body = md.convert(md_content)

# Create full HTML with styling
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DOVA: When Agents Learn to Think Together</title>
</head>
<body>
    <header>
        <h1 class="title">DOVA: When Agents Learn to Think Together</h1>
        <p class="subtitle">Building an Agentic Fabric for Deep Research Automation</p>
        <p class="author">DOVA Team</p>
    </header>
    <main>
        {html_body}
    </main>
</body>
</html>
"""

# Elegant CSS styling
css = CSS(string="""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

@page {{
    size: A4;
    margin: 2cm 2.5cm;
    @top-center {{
        content: "DOVA: Agentic Fabric for Deep Research";
        font-size: 9pt;
        color: #64748b;
        font-family: 'Inter', sans-serif;
    }}
    @bottom-center {{
        content: counter(page);
        font-size: 9pt;
        color: #64748b;
        font-family: 'Inter', sans-serif;
    }}
}}

body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1e293b;
}}

header {{
    text-align: center;
    margin-bottom: 2em;
    padding-bottom: 1.5em;
    border-bottom: 2px solid #e2e8f0;
}}

.title {{
    font-size: 28pt;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.3em;
    letter-spacing: -0.02em;
}}

.subtitle {{
    font-size: 14pt;
    color: #475569;
    font-weight: 500;
    margin-bottom: 0.5em;
}}

.author {{
    font-size: 11pt;
    color: #64748b;
    font-style: italic;
}}

h1 {{
    font-size: 20pt;
    font-weight: 700;
    color: #0f172a;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    padding-bottom: 0.3em;
    border-bottom: 2px solid #6366f1;
}}

h2 {{
    font-size: 14pt;
    font-weight: 600;
    color: #1e40af;
    margin-top: 1.3em;
    margin-bottom: 0.4em;
}}

h3 {{
    font-size: 12pt;
    font-weight: 600;
    color: #334155;
    margin-top: 1em;
}}

p {{
    margin-bottom: 0.8em;
    text-align: justify;
}}

blockquote {{
    border-left: 4px solid #6366f1;
    padding-left: 1em;
    margin: 1em 0;
    color: #475569;
    font-style: italic;
    background: #f8fafc;
    padding: 0.8em 1em;
}}

code {{
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 9pt;
    background: #f1f5f9;
    padding: 0.15em 0.4em;
    border-radius: 4px;
    color: #7c3aed;
}}

pre {{
    background: #1e293b;
    color: #e2e8f0;
    padding: 1em;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 8.5pt;
    line-height: 1.5;
    margin: 1em 0;
}}

pre code {{
    background: none;
    color: #e2e8f0;
    padding: 0;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0;
    font-size: 10pt;
}}

th {{
    background: #6366f1;
    color: white;
    padding: 0.6em 0.8em;
    text-align: left;
    font-weight: 600;
}}

td {{
    padding: 0.5em 0.8em;
    border-bottom: 1px solid #e2e8f0;
}}

tr:nth-child(even) {{
    background: #f8fafc;
}}

strong {{
    font-weight: 600;
    color: #0f172a;
}}

em {{
    font-style: italic;
    color: #475569;
}}

hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 2em 0;
}}

ul, ol {{
    margin-bottom: 0.8em;
    padding-left: 1.5em;
}}

li {{
    margin-bottom: 0.3em;
}}

/* Special callout box styling */
.callout {{
    background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
    border: 1px solid #bfdbfe;
    border-left: 4px solid #3b82f6;
    padding: 1em;
    border-radius: 8px;
    margin: 1.5em 0;
}}

/* Key takeaways styling */
h1 + p > strong:first-child {{
    color: #6366f1;
}}
""")

# Generate PDF
output_path = Path(__file__).parent / "dova-agentic-fabric.pdf"
HTML(string=html_content).write_pdf(output_path, stylesheets=[css])

print(f"PDF generated: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
