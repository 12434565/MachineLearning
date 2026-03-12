import graphviz

dot = graphviz.Digraph(format="png")
dot.node("A", "Hello")
dot.node("B", "World")
dot.edge("A", "B")

print(dot.render("gv_test", cleanup=True))
# brew install graphviz