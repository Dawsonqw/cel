import os
template_file="template.txt"

def gen_target(node_type):
    if os.path.exists(f"{node_type.lower()}.hpp"):
        print(f"{node_type.lower()}.hpp already exists")
        return
    node_type_upper=node_type.upper()
    with open(template_file, 'r') as f:
        template = f.read()
        template = template.replace("BASE_PARSER_TEMPLATE_H", f"BASE_PARSER_{node_type_upper}_H")
        template = template.replace("TemplateNode", f"{node_type}Node")
        template = template.replace("Template_type", node_type)
        with open(f"{node_type.lower()}.hpp", 'w') as f:
            f.write(template)
        with open("nodes.hpp", 'r') as f:
            lines = f.readlines()
            lines.insert(-1, f'#include "{node_type.lower()}.hpp"\n')
            with open("nodes.hpp", 'w') as f:
                f.writelines(lines)
# Input
gen_target("Input")
# Output
gen_target("Output")
# initializer
gen_target("Initializer")
# Conv
gen_target("Conv")
# Relu
gen_target("Relu")
# MaxPool
gen_target("MaxPool")
# Add
gen_target("Add")
# GlobalAveragePool
gen_target("GlobalAveragePool")
# Flatten
gen_target("Flatten")
# Gemm
gen_target("Gemm")