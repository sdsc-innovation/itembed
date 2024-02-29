from glob import glob
import json
import os

import mkdocs_gen_files


def convert(input_file, output_file):
    """Convert Jupyter notebook into a MkDocs-friendly Markdown file.

    This is an attempt to fix the inconsistencies introduced by `mkdocs-jupyter`, since
    it relies on `nbconvert`.

    """

    # Load notebook JSON
    notebook = json.load(input_file)

    # Convert cells, one by one
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            assert "outputs" not in cell

            # Write Markdown as-is
            for line in cell["source"]:
                output_file.write(line)
            output_file.write("\n")
            output_file.write("\n")

            continue

        # Write source code as code snippet
        if cell["cell_type"] == "code":
            output_file.write(
                f'<div class="notebook-number notebook-input">In [{cell["execution_count"]}]:</div>\n'
            )
            output_file.write("```python\n")
            for line in cell["source"]:
                output_file.write(line)
            output_file.write("\n")
            output_file.write("```\n")
            output_file.write("\n")

            def handle_output(output):
                # Treat streams (both stdout and stderr) as code output
                if output["output_type"] == "stream":
                    assert "data" not in output
                    output_file.write(
                        f'<div class="notebook-number notebook-output">Out [{cell["execution_count"]}]:</div>\n'
                    )
                    output_file.write("```\n")
                    for line in output["text"]:
                        output_file.write(line)
                    output_file.write("\n")
                    output_file.write("```\n")
                    output_file.write("\n")
                    return

                assert output["output_type"] in {"display_data", "execute_result"}

                data = output["data"]

                # Keep HTML as-is
                if "text/html" in data:
                    output_file.write(
                        f'<div class="notebook-number notebook-output">Out [{cell["execution_count"]}]:</div>\n'
                    )
                    output_file.write("<div>\n")
                    for line in data["text/html"]:
                        output_file.write(line)
                    output_file.write("\n")
                    output_file.write("</div>\n")
                    output_file.write("\n")
                    return

                # Wrap Javascript as script
                if "application/javascript" in data:
                    output_file.write("<script>\n")
                    for line in data["application/javascript"]:
                        output_file.write(line)
                    output_file.write("\n")
                    output_file.write("</script>\n")
                    output_file.write("\n")
                    return

                # Render plain text
                if "text/plain" in data:
                    output_file.write(
                        f'<div class="notebook-number notebook-output">Out [{cell["execution_count"]}]:</div>\n'
                    )
                    output_file.write("```\n")
                    for line in data["text/plain"]:
                        output_file.write(line)
                    output_file.write("\n")
                    output_file.write("```\n")
                    output_file.write("\n")
                    return

                # Throw, if output could not be converted
                raise NotImplementedError

            # A cell may have multiple outputs, just convert them sequentially
            for output in cell["outputs"]:
                handle_output(output)

            continue

        # Throw, in case there is an unexpected cell type
        raise NotImplementedError


for input_path in glob("docs/notebooks/*.ipynb"):
    name, _ = os.path.splitext(os.path.basename(input_path))
    output_path = f"notebooks/{name}.md"

    with open(input_path, "r", encoding="utf-8") as input_file:
        with mkdocs_gen_files.open(output_path, "w", encoding="utf-8") as output_file:
            print(input_path, output_path)
            convert(input_file, output_file)

    mkdocs_gen_files.set_edit_path(output_path, input_path[5:])
