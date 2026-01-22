import csv

def write_dict_to_csv(stats, name):
    """
    Given a dictionary of stats, organize it into a csv file for export.
    Assume stats is a dict of dict where the keys of the outer dict are the metrics 
    and the keys of the inner dict are the per-cam/per-time metric 
    """
    with open(f"{name}.csv", "w", newline='') as f:
        csv_writer = csv.writer(f)
        
        # Write header
        csv_writer.writerow(['Metric', 'Variant', 'Value'])
        
        # Iterate through metrics (psnr, lpips, ssim, chamfer, etc.)
        for metric, variants in stats.items():
            # Handle special case for chamfer if it has different structure
            if metric == "chamfer" and not isinstance(variants, dict):
                # If chamfer is just a single value, not a dict of variants
                csv_writer.writerow([metric, "", variants])
            else:
                # Iterate through each variant (n0_t0, n0_t1, etc.)
                for variant, value in variants.items():
                    csv_writer.writerow([metric, variant, value])