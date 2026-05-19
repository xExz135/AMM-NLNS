import argparse
import csv
import pickle
import os
from collections import defaultdict


def parse_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_int(value, default=None):
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def normalize_coordinates(instances):
    all_x = []
    all_y = []
    for depot, loc, _, _ in instances:
        all_x.append(depot[0])
        all_y.append(depot[1])
        for x, y in loc:
            all_x.append(x)
            all_y.append(y)
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        max_y = min_y + 1.0

    normalized = []
    for depot, loc, demand, capacity in instances:
        norm_depot = [
            (depot[0] - min_x) / (max_x - min_x),
            (depot[1] - min_y) / (max_y - min_y)
        ]
        norm_loc = [
            [ (x - min_x) / (max_x - min_x), (y - min_y) / (max_y - min_y) ]
            for x, y in loc
        ]
        normalized.append((norm_depot, norm_loc, demand, capacity))
    return normalized


def load_csv_instances(csv_path,
                       depot_lat_col='Restaurant_latitude',
                       depot_lon_col='Restaurant_longitude',
                       cust_lat_col='Delivery_location_latitude',
                       cust_lon_col='Delivery_location_longitude',
                       demand_col=None,
                       group_by_cols=None,
                       capacity=10,
                       default_demand=1):
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    if group_by_cols is None:
        group_by_cols = [depot_lat_col, depot_lon_col]

    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(col, '').strip() for col in group_by_cols)
        if any(v == '' for v in key):
            continue
        groups[key].append(row)

    instances = []
    skipped = 0
    for key, group_rows in groups.items():
        # Use first row in group as depot coordinates
        depot_x = parse_float(group_rows[0].get(depot_lat_col))
        depot_y = parse_float(group_rows[0].get(depot_lon_col))
        if depot_x is None or depot_y is None:
            skipped += len(group_rows)
            continue

        locations = []
        demands = []
        for row in group_rows:
            cust_x = parse_float(row.get(cust_lat_col))
            cust_y = parse_float(row.get(cust_lon_col))
            if cust_x is None or cust_y is None:
                skipped += 1
                continue
            demand = None
            if demand_col is not None and row.get(demand_col, '') != '':
                demand = parse_int(row.get(demand_col))
            if demand is None and 'multiple_deliveries' in row:
                demand = parse_int(row.get('multiple_deliveries'))
            if demand is None:
                demand = default_demand
            if demand <= 0:
                demand = default_demand

            locations.append([cust_x, cust_y])
            demands.append(demand)

        if not locations:
            continue

        instances.append(([depot_x, depot_y], locations, demands, capacity))

    return instances, skipped


def main():
    parser = argparse.ArgumentParser(
        description='Convert a delivery CSV into a VRP .pkl instance set for NLNS.'
    )
    parser.add_argument('--csv', required=True, help='Path to the source CSV file')
    parser.add_argument('--output', required=True, help='Output .pkl file path')
    parser.add_argument('--capacity', type=int, default=10, help='Vehicle capacity for each instance')
    parser.add_argument('--demand-column', type=str, default=None,
                        help='CSV column name for demand values. If omitted, uses multiple_deliveries or 1.')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize all coordinates to the range [0,1]')
    parser.add_argument('--group-by', type=str, default=None,
                        help='Comma-separated CSV columns to group orders into instances. Default groups by restaurant coords.')
    parser.add_argument('--depot-lat-col', default='Restaurant_latitude', help='CSV column for depot latitude')
    parser.add_argument('--depot-lon-col', default='Restaurant_longitude', help='CSV column for depot longitude')
    parser.add_argument('--cust-lat-col', default='Delivery_location_latitude',
                        help='CSV column for customer latitude')
    parser.add_argument('--cust-lon-col', default='Delivery_location_longitude',
                        help='CSV column for customer longitude')
    parser.add_argument('--default-demand', type=int, default=1,
                        help='Demand value for rows with no demand column')

    args = parser.parse_args()

    if args.group_by is None:
        group_by_cols = [args.depot_lat_col, args.depot_lon_col]
    else:
        group_by_cols = [col.strip() for col in args.group_by.split(',') if col.strip()]

    instances, skipped = load_csv_instances(
        args.csv,
        depot_lat_col=args.depot_lat_col,
        depot_lon_col=args.depot_lon_col,
        cust_lat_col=args.cust_lat_col,
        cust_lon_col=args.cust_lon_col,
        demand_col=args.demand_column,
        group_by_cols=group_by_cols,
        capacity=args.capacity,
        default_demand=args.default_demand
    )

    if args.normalize:
        instances = normalize_coordinates(instances)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(instances, f)

    print(f"Saved {len(instances)} instances to {args.output}")
    print(f"Skipped {skipped} rows with missing coordinates or invalid data")


if __name__ == '__main__':
    main()
