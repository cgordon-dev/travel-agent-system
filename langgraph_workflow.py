"""
LangGraph Workflow Visualization 
for Multi-Agent Travel & Event Planning System
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# Import core state definitions (simplified for visualization)
class GraphState:
    """Simplified state for visualization purposes"""
    user_input: str = ""
    planning_needed: bool = False
    booking_needed: bool = False
    monitoring_needed: bool = False
    task_complete: bool = False


def create_workflow_visualization():
    """Create a visual representation of the agent workflow"""
    
    # Define the graph - this is just for visualization purposes
    # The actual implementation is in travel_agent_architecture.py
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("user_interaction", lambda x: x)
    workflow.add_node("planning", lambda x: x)
    workflow.add_node("booking", lambda x: x)
    workflow.add_node("monitoring", lambda x: x)
    
    # Define edges (connections between agents)
    workflow.add_edge("user_interaction", "planning", condition=lambda x: x.planning_needed)
    workflow.add_edge("planning", "booking", condition=lambda x: x.booking_needed)
    workflow.add_edge("booking", "monitoring", condition=lambda x: x.monitoring_needed)
    workflow.add_edge("monitoring", "user_interaction", condition=lambda x: not x.task_complete)
    workflow.add_edge("monitoring", END, condition=lambda x: x.task_complete)
    
    # Conditional returns to user interaction
    workflow.add_edge("planning", "user_interaction", 
                      condition=lambda x: not x.booking_needed)
    workflow.add_edge("booking", "user_interaction", 
                      condition=lambda x: not x.monitoring_needed)
    
    # Define entry point
    workflow.set_entry_point("user_interaction")
    
    # Compile the graph (necessary for visualization)
    compiled = workflow.compile()
    
    return workflow, compiled


def visualize_graph(graph: CompiledGraph, filename: str = "travel_agent_workflow.png"):
    """Generate a visualization of the workflow graph"""
    
    # Extract the NetworkX graph
    nx_graph = graph.graph
    
    # Create a new figure
    plt.figure(figsize=(12, 8))
    
    # Node positions
    pos = {
        "user_interaction": (0.5, 0.8),
        "planning": (0.2, 0.5),
        "booking": (0.5, 0.2),
        "monitoring": (0.8, 0.5),
        "__end__": (1.0, 0.8)
    }
    
    # Node colors
    node_colors = {
        "user_interaction": to_rgba("royalblue", 0.8),
        "planning": to_rgba("forestgreen", 0.8),
        "booking": to_rgba("darkorange", 0.8),
        "monitoring": to_rgba("purple", 0.8),
        "__end__": to_rgba("crimson", 0.8)
    }
    
    # Node labels
    node_labels = {
        "user_interaction": "User Interaction\nAgent",
        "planning": "Planning\nAgent",
        "booking": "Booking\nAgent",
        "monitoring": "Monitoring\nAgent",
        "__end__": "End"
    }
    
    # Draw nodes
    for node in nx_graph.nodes():
        if node in pos:
            nx.draw_networkx_nodes(
                nx_graph, pos, 
                nodelist=[node], 
                node_color=[node_colors.get(node, "gray")], 
                node_size=5000, 
                alpha=0.9
            )
    
    # Draw node labels
    nx.draw_networkx_labels(
        nx_graph, pos, 
        labels=node_labels, 
        font_size=10, 
        font_weight="bold",
        font_color="white"
    )
    
    # Edge styles based on conditions
    edge_styles = {
        ("user_interaction", "planning"): {
            "label": "planning needed",
            "style": "solid",
            "color": "forestgreen"
        },
        ("planning", "booking"): {
            "label": "booking needed",
            "style": "solid",
            "color": "darkorange"
        },
        ("booking", "monitoring"): {
            "label": "monitoring needed",
            "style": "solid",
            "color": "purple"
        },
        ("monitoring", "user_interaction"): {
            "label": "needs more input",
            "style": "dashed",
            "color": "royalblue"
        },
        ("monitoring", "__end__"): {
            "label": "task complete",
            "style": "solid",
            "color": "crimson"
        },
        ("planning", "user_interaction"): {
            "label": "no booking needed",
            "style": "dashed",
            "color": "gray"
        },
        ("booking", "user_interaction"): {
            "label": "no monitoring needed",
            "style": "dashed",
            "color": "gray"
        }
    }
    
    # Draw edges with styles
    for edge in nx_graph.edges():
        if edge in edge_styles:
            style = edge_styles[edge]
            if style["style"] == "dashed":
                style_dict = {"linestyle": "dashed"}
            else:
                style_dict = {"linestyle": "solid"}
                
            # Draw edge
            nx.draw_networkx_edges(
                nx_graph, pos,
                edgelist=[edge],
                width=2,
                edge_color=style["color"],
                style=style_dict["linestyle"],
                alpha=0.7,
                arrowsize=20,
                connectionstyle="arc3,rad=0.1"
            )
            
            # Calculate edge midpoint for label
            mid_x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
            mid_y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
            
            # Add some offset for better label placement
            offset_x = (pos[edge[1]][0] - pos[edge[0]][0]) * 0.1
            offset_y = (pos[edge[1]][1] - pos[edge[0]][1]) * 0.1
            
            # Draw edge label
            plt.text(
                mid_x + offset_x, mid_y + offset_y, 
                style["label"],
                fontsize=8,
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
            )
    
    # Add title and description
    plt.title("Multi-Agent Travel & Event Planning System\nLangGraph Workflow", fontsize=15, pad=20)
    plt.text(
        0.5, 0.01,
        "This workflow shows the LangGraph-orchestrated flow between specialized agents.\n"
        "Each agent has specific responsibilities that collectively enable intelligent travel planning.",
        ha='center', va='center', fontsize=10,
        transform=plt.gca().transAxes
    )
    
    # Remove axes
    plt.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Workflow visualization saved to {filename}")


if __name__ == "__main__":
    # Create and visualize the workflow graph
    workflow, compiled_graph = create_workflow_visualization()
    visualize_graph(compiled_graph, "travel_agent_workflow.png")