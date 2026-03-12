import graphviz

# -----------------------------
# 1) Inputs (the day to predict)
# -----------------------------
x = {"temp_2": 39, "temp_1": 35, "average": 44, "friend": 30}

# --------------------------------------------------------
# 2) Manual traversal of the PROVIDED tree (homework tree)
#    This guarantees the expected answer: 41.0
# --------------------------------------------------------
def traverse_homework_tree(x):
    path = []  # store visited node IDs
    # Node IDs defined below in the Graphviz construction

    # root: temp_1 <= 59.5
    path.append("n0")
    if x["temp_1"] <= 59.5:
        # left: average <= 46.8
        path.append("n1")
        if x["average"] <= 46.8:
            # left: temp_1 <= 44.5
            path.append("n3")
            if x["temp_1"] <= 44.5:
                path.append("l0")  # leaf value 41.0
                return 41.0, ["temp_1", "average"], path
            else:
                path.append("l1")  # leaf value 45.0
                return 45.0, ["temp_1", "average"], path
        else:
            # In the provided diagram, this side continues via temp_1 <= 55.5 etc.
            # Not needed for your sample, but kept as placeholders.
            path.append("n4")
            if x["temp_1"] <= 55.5:
                path.append("l2")  # 51.9
                return 51.9, ["temp_1", "average"], path
            else:
                path.append("l3")  # 58.2
                return 58.2, ["temp_1", "average"], path
    else:
        # right: temp_1 <= 67.5
        path.append("n2")
        if x["temp_1"] <= 67.5:
            # average <= 60.8
            path.append("n5")
            if x["average"] <= 60.8:
                path.append("l4")  # 60.7
                return 60.7, ["temp_1", "average"], path
            else:
                path.append("l5")  # 66.3
                return 66.3, ["temp_1", "average"], path
        else:
            # average <= 75.6
            path.append("n6")
            if x["average"] <= 75.6:
                path.append("l6")  # 73.0
                return 73.0, ["temp_1", "average"], path
            else:
                path.append("l7")  # 80.6
                return 80.6, ["temp_1", "average"], path


pred, used_vars, path_nodes = traverse_homework_tree(x)

print("Prediction (homework tree):", pred)
print("Variables used:", used_vars)
print("Decision path node IDs:", path_nodes)

# ------------------------------------------
# 3) Build the tree diagram with Graphviz
#    and highlight the decision path in red
# ------------------------------------------
dot = graphviz.Digraph("tree", format="png")
dot.attr(rankdir="TB")

def add_node(node_id, label, on_path=False, shape="box"):
    if on_path:
        dot.node(node_id, label=label, shape=shape, style="filled", fillcolor="mistyrose", color="red", penwidth="2")
    else:
        dot.node(node_id, label=label, shape=shape, style="rounded,filled", fillcolor="white", color="black")

def add_edge(a, b, label):
    dot.edge(a, b, label=label)

on_path = set(path_nodes)

# Internal nodes (matching your provided diagram structure)
add_node("n0", "temp_1 <= 59.5\nmse=145.7\nsamples=162\nvalue=62.7", on_path=("n0" in on_path))
add_node("n1", "average <= 46.8\nmse=42.6\nsamples=63\nvalue=51.2", on_path=("n1" in on_path))
add_node("n2", "temp_1 <= 67.5\nmse=66.8\nsamples=99\nvalue=70.4", on_path=("n2" in on_path))

add_node("n3", "temp_1 <= 44.5\nmse=17.0\nsamples=17\nvalue=42.9", on_path=("n3" in on_path))
add_node("n4", "temp_1 <= 55.5\nmse=19.5\nsamples=46\nvalue=54.1", on_path=("n4" in on_path))

add_node("n5", "average <= 60.8\nmse=23.5\nsamples=42\nvalue=63.9", on_path=("n5" in on_path))
add_node("n6", "average <= 75.6\nmse=44.2\nsamples=57\nvalue=75.3", on_path=("n6" in on_path))

# Leaves
add_node("l0", "LEAF\nmse=4.4\nsamples=8\nvalue=41.0", on_path=("l0" in on_path), shape="box")
add_node("l1", "LEAF\nmse=22.2\nsamples=9\nvalue=45.0", on_path=("l1" in on_path), shape="box")
add_node("l2", "LEAF\nmse=7.7\nsamples=29\nvalue=51.9", on_path=("l2" in on_path), shape="box")
add_node("l3", "LEAF\nmse=15.6\nsamples=17\nvalue=58.2", on_path=("l3" in on_path), shape="box")
add_node("l4", "LEAF\nmse=13.7\nsamples=19\nvalue=60.7", on_path=("l4" in on_path), shape="box")
add_node("l5", "LEAF\nmse=17.3\nsamples=23\nvalue=66.3", on_path=("l5" in on_path), shape="box")
add_node("l6", "LEAF\nmse=34.3\nsamples=42\nvalue=73.0", on_path=("l6" in on_path), shape="box")
add_node("l7", "LEAF\nmse=27.1\nsamples=15\nvalue=80.6", on_path=("l7" in on_path), shape="box")

# Edges
add_edge("n0", "n1", "True")
add_edge("n0", "n2", "False")

add_edge("n1", "n3", "True")
add_edge("n1", "n4", "False")

add_edge("n3", "l0", "True")
add_edge("n3", "l1", "False")

add_edge("n4", "l2", "True")
add_edge("n4", "l3", "False")

add_edge("n2", "n5", "True")
add_edge("n2", "n6", "False")

add_edge("n5", "l4", "True")
add_edge("n5", "l5", "False")

add_edge("n6", "l6", "True")
add_edge("n6", "l7", "False")

# Render
out = dot.render("decision_path_graphviz", cleanup=True)
print("Saved path-highlighted tree to:", out)