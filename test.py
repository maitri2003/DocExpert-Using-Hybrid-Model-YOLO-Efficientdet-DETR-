import cv2
import matplotlib.pyplot as plt
import networkx as nx
from ultralytics import YOLO

# Load YOLOv9 model
model = YOLO('best.pt')  # Ensure 'best.pt' is in the current directory or provide the full path

# Load an image
image_path = r'C:\Users\SidMane\Documents\ML_Tutorials\ML_Programs\Projects\DocMap\main\DocExChanges\static\uploads\x-ray.png'  # Update with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

# Run inference
results = model(image_rgb)

# Extracting tensor data for visualization
cls = results[0].boxes.cls
conf = results[0].boxes.conf
xyxy = results[0].boxes.xyxy

# Initialize the graph
g = nx.DiGraph()

# Add nodes for the layers with additional tensor information
g.add_node("Input", pos=(0, 3), layer="Input Layer", tensor="Shape: (640, 288)")
g.add_node("YOLO", pos=(1, 2), layer="YOLO Detection", tensor=f"cls: {cls[0].tolist()}\nconf: {conf[0].tolist()}")
g.add_node("EfficientDet", pos=(1, 1), layer="EfficientDet Detection", tensor=f"cls: {cls[1].tolist()}\nconf: {conf[1].tolist()}")
g.add_node("DART", pos=(1, 0), layer="DART Classification", tensor=f"cls: {cls[2].tolist()}\nconf: {conf[2].tolist()}")
g.add_node("Output", pos=(2, 1), layer="Output Layer", tensor="Output predictions")

# Add edges to represent flow between layers
g.add_edge("Input", "YOLO")
g.add_edge("YOLO", "EfficientDet")
g.add_edge("EfficientDet", "DART")
g.add_edge("DART", "Output")

# Define positions for visualization
pos = nx.get_node_attributes(g, 'pos')

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw(
    g, 
    pos, 
    with_labels=True, 
    node_size=2000, 
    node_color=["lightblue", "lightgreen", "yellow", "orange", "red"],
    font_size=10, 
    font_weight="bold",
    arrows=True, 
    arrowsize=20
)

# Annotate the layers with tensor info
for node in g.nodes:
    layer_tensor_info = g.nodes[node]['tensor']
    x_pos, y_pos = pos[node]
    plt.text(x_pos, y_pos - 0.2, layer_tensor_info, fontsize=9, ha='center')

# Add title and save the graph
plt.title("Hybrid Model Layer Visualization with Tensor Information", fontsize=14)
plt.savefig("hybrid_model_layers_with_tensor.png")
plt.show()
