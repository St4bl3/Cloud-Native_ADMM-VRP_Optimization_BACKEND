import logging
import os
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import uvicorn
import asyncio
import httpx
from typing import List, Dict
import polyline  # For decoding polyline strings

# Initialize FastAPI app
app = FastAPI()

# Configure CORS (Adjust allow_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local testing
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Configure Logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
logger = logging.getLogger("vrp-api")

# Environment Variables
SUPABASE_URL = "https://wqabbdjzrentixadsaff.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndxYWJiZGp6cmVudGl4YWRzYWZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE5MzA1MDksImV4cCI6MjA0NzUwNjUwOX0.K2rv84ukth3maDE0Zn7gLBae9qcHNnidcIKNZrCgRYk"
ORS_API_KEY ='5b3ce3597851110001cf6248cea1a7cb28584eb093979bcf3280480a'

# Validate Environment Variables
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    logger.error("Supabase credentials are not set. Please check your environment variables.")
    raise EnvironmentError("Supabase credentials are missing.")

if not ORS_API_KEY:
    logger.error("ORS_API_KEY is not set. Please check your environment variables.")
    raise EnvironmentError("ORS_API_KEY is missing.")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ORS Configuration
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"

# In-memory store for assignments
current_assignments = {
    "routes": []  # Stores assignments from /solve
}

# Initialize an asyncio Lock for thread-safe access to current_assignments
assignment_lock = asyncio.Lock()


# Utility Functions

def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Calculate a symmetric distance matrix based on coordinates.
    """
    return distance_matrix(coords, coords)


def get_coordinates(warehouses_df: pd.DataFrame, warehouse_id: int) -> List[float]:
    """
    Retrieve X and Y coordinates for a given WarehouseID.
    Returns [longitude, latitude]
    """
    warehouse = warehouses_df[warehouses_df['WarehouseID'] == warehouse_id]
    if warehouse.empty:
        raise ValueError(f"WarehouseID {warehouse_id} not found.")
    # Return [longitude, latitude]
    return warehouse.iloc[0][['X', 'Y']].tolist()


def simplify_route(coordinates: List[List[float]], max_points=100) -> List[List[float]]:
    """
    Simplify the route by reducing the number of points.

    Args:
        coordinates (List[List[float]]): The list of coordinates.
        max_points (int): The maximum number of points to include.

    Returns:
        List[List[float]]: The simplified list of coordinates.
    """
    if len(coordinates) <= max_points:
        return coordinates
    else:
        # Calculate sampling interval
        interval = len(coordinates) / max_points
        sampled_coords = [coordinates[int(i * interval)] for i in range(max_points)]
        return sampled_coords


async def fetch_route_async(source_coords: List[float], dest_coords: List[float], client: httpx.AsyncClient) -> List[List[float]]:
    """
    Asynchronously fetch route from ORS between source and destination coordinates.

    Implements retry logic and handles rate limiting gracefully.
    """
    headers = {
        'Authorization': ORS_API_KEY,
        'Content-Type': 'application/json'
    }
    body = {
        "coordinates": [
            source_coords,  # ORS expects [longitude, latitude]
            dest_coords
        ],
        "instructions": False  # We don't need turn-by-turn instructions
    }
    logger.debug(f"Sending async request to ORS: {body}")
    max_retries = 2
    backoff_factor = 1  # Adjust backoff as needed
    for attempt in range(max_retries):
        try:
            response = await client.post(ORS_URL, json=body, headers=headers, timeout=10)
            logger.debug(f"Received async response from ORS (status {response.status_code}): {response.text}")
            if response.status_code == 429:
                # Rate limit exceeded
                retry_after = int(response.headers.get('Retry-After', '1'))
                logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                await asyncio.sleep(retry_after + backoff_factor * attempt)
                continue
            elif response.status_code != 200:
                try:
                    error_info = response.json()
                    error_message = error_info.get('error', {}).get('message', response.text)
                except ValueError:
                    error_message = response.text
                logger.error(f"ORS API Error {response.status_code}: {error_message}")
                return None  # Return None instead of raising exception
            else:
                break  # Successful response
        except httpx.RequestError as e:
            logger.error(f"Async request to ORS failed: {e}")
            return None  # Return None instead of raising exception

    if response.status_code != 200:
        logger.error(f"Failed to get route after {max_retries} attempts.")
        return None  # Return None

    try:
        data = response.json()
    except ValueError:
        logger.error("ORS response is not valid JSON.")
        return None  # Return None

    if 'routes' not in data:
        logger.error(f"ORS response missing 'routes': {data}")
        return None

    if not data['routes']:
        logger.error(f"ORS 'routes' is empty: {data}")
        return None

    try:
        # Extract and decode the geometry polyline
        encoded_polyline = data['routes'][0]['geometry']
        coordinates = polyline.decode(encoded_polyline)
        logger.debug(f"Decoded coordinates (lat, lon): {coordinates}")

        # Reverse each coordinate pair to [longitude, latitude]
        coordinates = [[lon, lat] for lat, lon in coordinates]
        logger.debug(f"Reversed coordinates (lon, lat): {coordinates}")

        # Simplify the route if needed
        simplified_coordinates = simplify_route(coordinates, max_points=100)
        return simplified_coordinates
    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"Error extracting coordinates from ORS response: {e}")
        return None


def assign_demands_admm(trucks: pd.DataFrame, demands: pd.DataFrame, dist_matrix: np.ndarray) -> np.ndarray:
    """
    Perform demand assignment using ADMM.
    """
    num_trucks = len(trucks)
    num_demands = len(demands)
    z = np.zeros((num_trucks, num_demands))  # Assignment matrix
    rho = 1.0  # Penalty parameter
    max_iter = 100  # Maximum iterations
    tolerance = 1e-4  # Convergence tolerance

    priorities = demands["Priority"].apply(lambda x: 1 if str(x).lower() == "high" else 0).to_numpy()
    norm_priorities = priorities / (priorities.sum() or 1)
    quantities = demands["Quantity"].to_numpy()
    norm_quantities = quantities / (quantities.sum() or 1)
    max_dist = np.max(dist_matrix) or 1
    norm_dist_matrix = dist_matrix / max_dist

    for iteration in range(max_iter):
        z_old = z.copy()
        for i in range(num_trucks):
            truck_capacity = trucks.loc[i, "Capacity"] - trucks.loc[i, "CurrentLoad"]
            for j in range(num_demands):
                if (
                    demands.loc[j, "ACRequired"] == trucks.loc[i, "AC"]
                    and demands.loc[j, "Quantity"] <= truck_capacity
                    and str(trucks.loc[i, "Status"]).lower() == "available"
                ):
                    cost_term = norm_quantities[j] * 0.4
                    priority_term = norm_priorities[j] * 0.3
                    distance_term = norm_dist_matrix[
                        trucks.loc[i, "CurrentWarehouseID"] - 1,
                        demands.loc[j, "SourceID"] - 1,
                    ] * 0.3
                    z[i, j] = max(0, 1 - (cost_term + priority_term + distance_term))
                    truck_capacity -= demands.loc[j, "Quantity"]
                else:
                    z[i, j] = 0

        if np.linalg.norm(z - z_old) < tolerance:
            logger.info(f"Convergence achieved after iteration: {iteration}")
            break
    else:
        logger.warning("ADMM did not converge within the maximum number of iterations.")

    return z


def solve_vrp(assigned_matrix: np.ndarray, demands: pd.DataFrame) -> List[Dict]:
    """
    Solve VRP and output demand assignments for each truck.
    """
    routes = []
    for truck_idx, assignments in enumerate(assigned_matrix):
        truck_id = truck_idx + 1
        assigned_demands = [
            demands.iloc[demand_idx].to_dict()
            for demand_idx, assigned in enumerate(assignments)
            if assigned > 0
        ]
        routes.append({"truck_id": truck_id, "assigned_demands": assigned_demands})
    return routes


async def fetch_truck_path(route: Dict, warehouses_df: pd.DataFrame, trucks_df: pd.DataFrame, client: httpx.AsyncClient) -> Dict:
    """
    Fetch path for a single truck asynchronously.
    """
    truck_id = route["truck_id"]
    assigned_demands = route["assigned_demands"]
    if not assigned_demands:
        return {
            "truck_id": truck_id,
            "path": []
        }

    # Get truck details
    truck = trucks_df[trucks_df["TruckID"] == truck_id]
    if truck.empty:
        logger.error(f"Truck ID {truck_id} not found.")
        return {
            "truck_id": truck_id,
            "path": []
        }
    truck = truck.iloc[0]
    try:
        current_location = get_coordinates(warehouses_df, truck['CurrentWarehouseID'])
    except ValueError as e:
        logger.error(f"Error fetching coordinates for WarehouseID {truck['CurrentWarehouseID']}: {e}")
        return {
            "truck_id": truck_id,
            "path": []
        }

    path = []
    total_route = []

    for demand in assigned_demands:
        try:
            source = get_coordinates(warehouses_df, demand["SourceID"])
            destination = get_coordinates(warehouses_df, demand["DestinationID"])
        except ValueError as e:
            logger.error(f"Error fetching coordinates for DemandID {demand['DemandID']}: {e}")
            continue

        # Route from current location to source (pickup)
        if current_location != source:
            pickup_route = await fetch_route_async(current_location, source, client)
            if pickup_route:
                total_route.extend(pickup_route[:-1])  # Exclude last point to avoid duplicates
            else:
                logger.error(f"Failed to fetch pickup route for Truck {truck_id}.")
                continue
            current_location = source

        # Route from source to destination (delivery)
        delivery_route = await fetch_route_async(current_location, destination, client)
        if delivery_route:
            total_route.extend(delivery_route[:-1])  # Exclude last point to avoid duplicates
        else:
            logger.error(f"Failed to fetch delivery route for Truck {truck_id}.")
            continue
        current_location = destination

    # Append the last location
    total_route.append(current_location)

    return {
        "truck_id": truck_id,
        "path": total_route  # List of [longitude, latitude]
    }


async def fetch_paths(routes: List[Dict], warehouses_df: pd.DataFrame, trucks_df: pd.DataFrame) -> List[Dict]:
    """
    Fetch paths for all trucks asynchronously based on their assigned demands.
    """
    truck_paths = []
    semaphore = asyncio.Semaphore(10)  # Increased concurrency limit to handle more trucks

    async with httpx.AsyncClient() as client:
        async def limited_fetch_truck_path(route):
            async with semaphore:
                return await fetch_truck_path(route, warehouses_df, trucks_df, client)

        # Process all trucks
        tasks = [
            limited_fetch_truck_path(route)
            for route in routes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Error fetching truck path: {result}")
            continue
        if result is not None:
            truck_paths.append(result)

    return truck_paths


# API Endpoints

@app.get("/")
def home():
    """
    Home endpoint to confirm the API is running.
    """
    return {"message": "Welcome to the VRP API!"}


@app.post("/solve")
async def solve_vrp_api():
    """
    Endpoint to solve the VRP using data from Supabase.

    Assigns demands to trucks using ADMM and stores the assignment in memory.
    """
    async with assignment_lock:
        try:
            # Fetch data from Supabase
            warehouses = supabase.table("warehouses").select("*").execute()
            trucks = supabase.table("trucks").select("*").execute()
            demands = supabase.table("demands").select("*").execute()

            if not warehouses.data or not trucks.data or not demands.data:
                raise HTTPException(status_code=400, detail="One or more tables are empty.")

            # Convert data to Pandas DataFrame
            warehouses_df = pd.DataFrame(warehouses.data)
            trucks_df = pd.DataFrame(trucks.data)
            demands_df = pd.DataFrame(demands.data)

            # Calculate distance matrix
            coords = warehouses_df[["X", "Y"]].to_numpy()
            dist_matrix = calculate_distance_matrix(coords)

            # Perform demand assignment using ADMM
            assigned_matrix = assign_demands_admm(trucks_df, demands_df, dist_matrix)

            # Solve VRP based on the assigned demands
            routes = solve_vrp(assigned_matrix, demands_df)

            # Update the in-memory assignments
            current_assignments["routes"] = routes

            logger.info("Demand assignment completed successfully.")

            # Return the assignments as JSON
            return {"routes": routes}

        except HTTPException as e:
            logger.error(f"HTTPException in /solve: {e.detail}")
            raise e
        except Exception as e:
            logger.exception("Unexpected error in /solve.")
            raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/routing")
async def get_routing():
    """
    Endpoint to get the path taken by all trucks based on current assignments.

    Fetches routes from ORS and returns the paths.
    """
    async with assignment_lock:
        try:
            # Check if there are current assignments
            if not current_assignments["routes"]:
                raise HTTPException(status_code=400, detail="No current assignments. Please call /solve first.")

            # Fetch data from Supabase
            warehouses = supabase.table("warehouses").select("*").execute()
            trucks = supabase.table("trucks").select("*").execute()
            demands = supabase.table("demands").select("*").execute()

            if not warehouses.data or not trucks.data or not demands.data:
                raise HTTPException(status_code=400, detail="One or more tables are empty.")

            # Convert data to Pandas DataFrame
            warehouses_df = pd.DataFrame(warehouses.data)
            trucks_df = pd.DataFrame(trucks.data)
            demands_df = pd.DataFrame(demands.data)

            # Fetch paths based on current assignments
            truck_paths = await fetch_paths(current_assignments["routes"], warehouses_df, trucks_df)

            logger.info("Routing information fetched successfully.")

            # Return the paths as JSON
            return {"truck_paths": truck_paths}

        except HTTPException as e:
            logger.error(f"HTTPException in /routing: {e.detail}")
            raise e
        except Exception as e:
            logger.exception("Unexpected error in /routing.")
            raise HTTPException(status_code=500, detail="Internal Server Error")
# Entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)