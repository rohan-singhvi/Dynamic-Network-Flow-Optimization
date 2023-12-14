import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point


def load_data(file_path):
    """Load and preprocess traffic data from a CSV file."""
    df = pd.read_csv(file_path)
    # Fill NaN values with the day average
    for col in df.columns:
        if col.startswith('hour_'):
            df[col].fillna(df['day_avg'], inplace=True)
    return df

def build_network_graph(geojson_path):
    """Build a network graph from the geojson file."""
    geo_df = gpd.read_file(geojson_path)
    G = nx.DiGraph()
    for index, row in geo_df.iterrows():
        # Assuming each MultiLineString's first point is the start and last is the end
        if isinstance(row['geometry'], Point):
            continue
        start_point = row['geometry'][0].coords[0]
        end_point = row['geometry'][-1].coords[-1]
        segment_id = row['segment_id']
        G.add_node(start_point)
        G.add_node(end_point)
        G.add_edge(start_point, end_point, segment_id=segment_id)
    return G

def define_variables(model, data, G):
    """Define the binary decision variables for the optimization model."""
    segments = data['segment_id'].unique()
    hours = [f'hour_{h}_avg' for h in range(24)]
    x_vars = model.addVars(segments, hours, vtype=GRB.BINARY, name="x")
    return x_vars

def add_objective_function(model, x_vars, data):
    """Add the objective function to minimize total travel time."""
    objective = gp.quicksum(x_vars[segment, hour] * data.at[segment, hour]
                            for segment in x_vars.keys()
                            for hour in x_vars[segment].keys())
    model.setObjective(objective, GRB.MINIMIZE)

def add_constraints(model, x_vars, data, G):
    """Add constraints to the model based on the network graph."""
    time_periods = 24  # This should be set according to your problem's requirements
    origin = 'origin_node_identifier'  # Replace with the actual identifier for the origin node
    destination = 'destination_node_identifier'  # Replace with the actual identifier for the destination node
    num_segments = len(data['segment_id'].unique())
    segment = data['segment_id'].unique()
    # Connectivity constraints
    for n in G.nodes():
        if n not in [origin, destination]:
            for t in range(time_periods):
                model.addConstr(
                    gp.quicksum(x_vars[i, t] for i, _ in G.in_edges(n)) ==
                    gp.quicksum(x_vars[j, t] for _, j in G.out_edges(n))
                )

    for t in range(time_periods):
        model.addConstr(gp.quicksum(x_vars[i, t] for i in G.in_edges(origin)) == 0)
        model.addConstr(gp.quicksum(x_vars[j, t] for j in G.out_edges(destination)) == 0)

    # Add path continuity constraints
    for i in range(num_segments):
        for t in range(time_periods - 1):
            model.addConstr(x_vars[i, t] <= gp.quicksum(x_vars[j, t + 1] for j in segment[i]))


def optimize_route(traffic_data_path, network_data_path):
    """Optimize the route based on traffic and network data."""
    traffic_data = load_data(traffic_data_path)
    network_graph = build_network_graph(network_data_path)

    # Create a new model
    model = gp.Model("RouteOptimization")

    # Define variables
    x_vars = define_variables(model, traffic_data, network_graph)

    # Add objective function
    add_objective_function(model, x_vars, traffic_data)

    # Add constraints
    add_constraints(model, x_vars, traffic_data, network_graph)

    # Optimize model
    model.optimize()

    # Extract the solution
    solution = {}
    for v in model.getVars():
        if v.x > 0.5:  # Only consider segments chosen in the solution
            solution[v.varName] = v.x
    return solution

if __name__ == "__main__":
    traffic_data_path = './week_2016-01-11/summary_week_2016-01-11.csv'
    network_data_path = 'kl_segments.geojson'
    solution = optimize_route(traffic_data_path, network_data_path)
    print(solution)
