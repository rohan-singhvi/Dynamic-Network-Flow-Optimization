import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import geopandas as gpd
import random
import networkx as nx
from shapely.geometry import Point, MultiLineString


def load_data(file_path):
    """Load and preprocess traffic data from a CSV file."""
    df = pd.read_csv(file_path)
    # Fill NaN values with the day average
    for col in df.columns:
        if col.startswith('hour_'):
            df[col].fillna(df['day_avg'], inplace=True)
    return df

def round_coords(coord):
    return (round(coord[0], 6), round(coord[1], 6))

def build_network_graph(geo_df):
    """Build a network graph from a GeoDataFrame."""
    G = nx.DiGraph()
    for index, row in geo_df.iterrows():
        geom = row['geometry']
        segment_id = row['segment_id']
        if isinstance(geom, Point):
            continue
        if isinstance(geom, MultiLineString):
            for line in geom.geoms:
                start_point = round_coords(line.coords[0])
                end_point = round_coords(line.coords[-1])
                G.add_edge(start_point, end_point, segment_id=segment_id)
        else:
            start_point = round_coords(geom.coords[0])
            end_point = round_coords(geom.coords[-1])
            G.add_edge(start_point, end_point, segment_id=segment_id)
    return G

def map_segments_to_edges(geo_df, G):
    """Map segment IDs to edges in the graph."""
    segment_to_edge = {}
    for index, row in geo_df.iterrows():
        segment_id = row['segment_id']
        geom = row['geometry']
        if isinstance(geom, MultiLineString):
            line = list(geom.geoms)[0]
        else:
            line = geom
        start_point = round_coords(line.coords[0])
        end_point = round_coords(line.coords[-1])
        segment_to_edge[segment_id] = (start_point, end_point)
    return segment_to_edge


def define_variables(model, data, G):
    """Define the binary decision variables for the optimization model."""
    hours = [f'hour_{h}_avg' for h in range(24)]
    x_vars = model.addVars(G.edges(), hours, vtype=GRB.BINARY, name="x")
    return x_vars

def add_objective_function(model, x_vars, data, segment_to_edge):
    """Add the objective function to minimize total travel time."""
    objective = gp.quicksum(
        x_vars[edge, hour] * data.loc[data['segment_id'] == segment, hour].values[0]
        for segment, edge in segment_to_edge.items()
        for hour in [f'hour_{h}_avg' for h in range(24)]
        if (edge, hour) in x_vars  # Ensure the key exists in x_vars
    )
    model.setObjective(objective, GRB.MINIMIZE)


def add_constraints(model, x_vars, data, G, segment_to_edge):
    """Add constraints to the model based on the network graph."""
    time_periods = [f'hour_{h}_avg' for h in range(24)]

    # Random origin and destination selection
    nodes = list(G.nodes())
    origin, destination = random.sample(nodes, 2)

    # Connectivity constraints
    for n in G.nodes():
        if n not in [origin, destination]:
            for t in time_periods:
                in_edges = [segment_to_edge[segment] for segment, edge in segment_to_edge.items() 
                            if edge[0] == n and (edge, t) in x_vars]
                out_edges = [segment_to_edge[segment] for segment, edge in segment_to_edge.items() 
                             if edge[1] == n and (edge, t) in x_vars]

                model.addConstr(gp.quicksum(x_vars[edge, t] for edge in in_edges) ==
                                gp.quicksum(x_vars[edge, t] for edge in out_edges))

    # Origin-destination constraints
    for t in time_periods:
        origin_out_edges = [segment_to_edge[segment] for segment, edge in segment_to_edge.items() 
                            if edge[0] == origin and (edge, t) in x_vars]
        destination_in_edges = [segment_to_edge[segment] for segment, edge in segment_to_edge.items() 
                                if edge[1] == destination and (edge, t) in x_vars]
        while origin == destination:
            origin, destination = random.sample(nodes, 2)
        model.addConstr(gp.quicksum(x_vars[edge, t] for edge in origin_out_edges) == -1)  # Origin has one outgoing edge
        model.addConstr(gp.quicksum(x_vars[edge, t] for edge in destination_in_edges) == 1)  # Destination has one incoming edge


def optimize_route(traffic_data_path, network_data_path):
    """Optimize the route based on traffic and network data."""
    traffic_data = load_data(traffic_data_path)
    network_data = gpd.read_file(network_data_path)

    # Filter datasets for common segment_ids
    common_segment_ids = set(traffic_data['segment_id']).intersection(set(network_data['segment_id']))
    traffic_data = traffic_data[traffic_data['segment_id'].isin(common_segment_ids)]
    network_data = network_data[network_data['segment_id'].isin(common_segment_ids)]

    network_graph = build_network_graph(network_data)
    segment_to_edge = map_segments_to_edges(network_data, network_graph)
    # Create and optimize the model
    model = gp.Model("RouteOptimization")
    x_vars = define_variables(model, traffic_data, network_graph)
    add_objective_function(model, x_vars, traffic_data, segment_to_edge)
    add_constraints(model, x_vars, traffic_data, network_graph, segment_to_edge)
    model.optimize()
    if model.status == GRB.INF_OR_UNBD or model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
        # Perform infeasibility analysis
        model.computeIIS()
        model.write("model.ilp")
        print("Infeasibility report written to 'model.ilp'")
        return None

    # Extract and return the solution if feasible
    if model.status == GRB.OPTIMAL:
        solution = {v.varName: v.x for v in model.getVars() if v.x > 0.5}
        return solution

    return None


if __name__ == "__main__":
    traffic_data_path = './week_2016-01-11/summary_week_2016-01-11.csv'
    network_data_path = 'kl_segments.geojson'
    solution = optimize_route(traffic_data_path, network_data_path)
    print(solution)
